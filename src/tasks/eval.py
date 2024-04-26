def get_predict_file(output_dir, args, data_yaml_file):
    cc = ['pred']
    # example data_yaml_file: datasets/coco_caption/test.yaml
    data = data_yaml_file.split('/')[-2]
    if data != 'coco_caption':
        cc.append(data)
    cc.append(op.splitext(op.basename(data_yaml_file))[0])
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))


def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    return op.splitext(predict_file)[0] + '.eval.json'



def test(args, test_dataloader, model, tokenizer, predict_file):

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        # local_rank would not work for cross-node distributed training
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(
            get_rank(), world_size) + op.splitext(predict_file)[1]

    model.eval()

    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        exist_key2pred = {}
        tmp_file = cache_file + '.tmp.copy'
        if op.isfile(tmp_file):
            with open(tmp_file, 'r') as fp:
                for line in fp:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        exist_key2pred[parts[0]] = parts[1]

        with torch.no_grad():
            for step, (img_keys, batch, meta_data) in tqdm(enumerate(test_dataloader)):
                # torch.cuda.empty_cache()
                is_exist = True
                for k in img_keys:
                    if k not in exist_key2pred:
                        is_exist = False
                        break
                if is_exist:
                    for k in img_keys:
                        yield k, exist_key2pred[k]
                    continue
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'is_decode': True,
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'img_feats': batch[3],
                    'audio_feat': batch[4],
                    'masked_pos': batch[5],
                    'input_token_ids': batch[6],
                    'output_token_ids': batch[7],
                    'do_sample': False,
                    'bos_token_id': cls_token_id,
                    'pad_token_id': pad_token_id,
                    'eos_token_ids': [sep_token_id],
                    'mask_token_id': mask_token_id,
                    # for adding od labels
                    'add_od_labels': args.add_od_labels,
                    'od_labels_start_posid': args.max_seq_a_length,
                    # hyperparameters of beam search
                    'max_length': args.max_gen_length,
                    'num_beams': args.num_beams,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "num_return_sequences": args.num_return_sequences,
                    "num_keep_best": args.num_keep_best,
                }

                tic = time.time()
                # captions, logprobs

                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(),
                                               skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    yield img_key, json.dumps(res)

        logger.info(
            f"Inference model computing time: {(time_meter / (step+1))} seconds per batch"
        )

    tsv_writer(gen_rows(), cache_file)
    if world_size > 1:
        dist.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
            op.splitext(predict_file)[1] for i in range(world_size)]
        concat_tsv_files(cache_files, predict_file)
        delete_tsv_files(cache_files)
        reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys,
                         predict_file)
    if world_size > 1:
        dist.barrier()


def evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir, args,
                                    val_dataloader.dataset.yaml_file)
    test(args, val_dataloader, model, tokenizer, predict_file)

    if get_world_size() > 1:
        dist.barrier()
    evaluate_file = get_evaluate_file(predict_file)
    if is_main_process():
        caption_file = val_dataloader.dataset.get_caption_file_in_coco_format()
        data = val_dataloader.dataset.yaml_file.split('/')[-2]
        result = evaluate_on_coco_caption(predict_file,
                                          caption_file,
                                          outfile=evaluate_file)
        logger.info(f'evaluation result: {str(result)}')
        logger.info(f'evaluation result saved to {evaluate_file}')
    if get_world_size() > 1:
        dist.barrier()
    return evaluate_file