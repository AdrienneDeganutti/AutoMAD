import io
import json
import os.path as op
from pathlib import Path

from PIL import Image
import av
import h5py
import numpy as np
from src.utils.load_files import (
    find_file_path_in_yaml,
    load_box_linelist_file,
    load_from_yaml_file,
)
from src.utils.logger import LOGGER
from src.utils.tsv_file import CompositeTSVFile, TSVFile
from src.utils.tsv_file_ops import tsv_reader
import torch
from torchvision import transforms
#from .data_utils.image_ops import img_from_base64
#from .data_utils.video_ops import extract_frames_from_video_path

#from .data_utils.volume_transforms import ClipToTensor


class VisionLanguageTSVDataset(object):

    def __init__(self,
                 args,
                 yaml_file,
                 tokenizer,
                 tensorizer=None,
                 is_train=None,
                 on_memory=False):

        self.args = args
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.yaml_file = yaml_file
        self.root = Path(op.dirname(yaml_file)).parent.absolute()

        self.cfg = load_from_yaml_file(yaml_file)
        self.is_composite = self.cfg.get('composite', False)
        self.cap_linelist_file = find_file_path_in_yaml(
            self.cfg.get('caption_linelist', None), op.join(self.root, 'metadata'))


        self.visual_file = self.cfg.get('img', None)
        self.visual_tsv = self.get_tsv_file(self.visual_file)

        self.label_file = self.cfg.get('label', None)
        self.label_tsv = self.get_tsv_file(op.join('metadata', self.label_file))

        self.cap_file = self.cfg.get('caption', None)
        self.cap_tsv = self.get_tsv_file(op.join('metadata', self.cap_file))

        if self.is_composite:
            assert op.isfile(self.cap_linelist_file)
            self.cap_line_list = [
                int(row[2]) for row in tsv_reader(self.cap_linelist_file)
            ]
            self.img_line_list = [i for i in range(len(self.cap_line_list))]

        # True
        elif self.cap_linelist_file:
            line_list = load_box_linelist_file(self.cap_linelist_file)
            self.img_line_list = line_list[0]
            self.cap_line_list = line_list[1]
        else:
            # one caption per image/video
            self.img_line_list = [i for i in range(self.label_tsv.num_rows())]
            self.cap_line_list = [0 for i in range(self.label_tsv.num_rows())]

        if is_train:
            assert self.cap_tsv is not None
            assert tokenizer is not None

        self.is_train = is_train
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.on_memory = on_memory
        if on_memory:
            if self.cap_tsv is not None:
                self.load_caption_to_memory()

        self.is_train = is_train
        #self.img_res = getattr(args, 'img_res', 224)
        #self.patch_size = getattr(args, 'patch_size', 16)

        self.img_feature_dim = args.img_feature_dim
        self.decoder_target_fps = 5
        self.decoder_num_frames = getattr(args, 'max_num_frames', 2)
        self.decoder_multi_thread_decode = False

        self.decoder_safeguard_duration = False
        self.add_od_labels = getattr(args, 'add_od_labels', False)
        self.use_asr = getattr(args, 'use_asr', False)

        LOGGER.info(f'Use_asr: {self.use_asr}')
        # use uniform sampling as default for now
        self.decoder_sampling_strategy = getattr(args,
                                                 'decoder_sampling_strategy',
                                                 'uniform')
        LOGGER.info(f'isTrainData: {self.is_train}\n[PyAV video parameters] '
                    f'Num of Frame: {self.decoder_num_frames}, '
                    f'FPS: {self.decoder_target_fps}, '
                    f'Sampling: {self.decoder_sampling_strategy}')


    def roll_func(self, x, axis=1, shift=None, shift_range=50):
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.randint(-shift_range, shift_range))

        return x.roll(sf, axis)

    def get_composite_source_idx(self):
        if self.is_composite:
            assert op.isfile(self.cap_linelist_file)
            self.composite_source_idx = [
                int(row[0]) for row in tsv_reader(self.cap_linelist_file)
            ]
        else:
            # only a single tsv file is used as input
            self.composite_source_idx = [
                0 for _ in range(len(self.cap_line_list))
            ]
        return self.composite_source_idx

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(tsv_file,
                                        self.cap_linelist_file,
                                        root=self.root)
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def load_caption_to_memory(self):
        self.caption_on_memory = {}
        for img_idx in set(self.img_line_list):
            row = self.get_row_from_tsv(self.cap_tsv, img_idx)
            for cap_idx, data in enumerate(json.loads(row[1])):
                self.caption_on_memory[(img_idx, cap_idx)] = data['caption']

    def get_valid_tsv(self):
        if self.is_train:
            return self.cap_tsv
        # sorted by file size
        if self.cap_tsv:
            return self.cap_tsv
        if self.visual_tsv:
            return self.visual_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.get_key(i) for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i): i for i in range(tsv.num_rows())}

    def get_image_cap_index(self, idx):
        return self.img_line_list[idx], self.cap_line_list[idx]

    def get_row_from_tsv(self, tsv, img_idx):
        row = tsv[img_idx]
        if self.is_composite:
            assert self.image_keys[img_idx].endswith(row[0])
        else:
            if row[0] != self.image_keys[img_idx]:
                print(row[0], self.image_keys[img_idx])
            assert row[0] == self.image_keys[img_idx]
        return row

    def get_caption(self, img_idx, cap_idx):
        if self.is_train:
            if self.on_memory:
                return self.caption_on_memory[(img_idx, cap_idx)]
            row = self.get_row_from_tsv(self.cap_tsv, img_idx)
            return json.loads(row[1])[cap_idx]['caption']
        return ""

    def get_caption_and_timeinfo(self, data, cap_idx):
        caption, tag, start, end = '', ' ', None, None
        data_sample = data[cap_idx]
        if self.is_train:
            caption = data_sample['caption']
            if 'start' in data_sample.keys():
                start = data_sample['start']
            if 'end' in data_sample.keys():
                end = data_sample['end']
            if 'asr' in data_sample.keys() and self.use_asr:
                asr = data_sample['asr'].lower()
                tag = asr
        else:
            if 'start' in data_sample.keys():
                start = data_sample['start']
            if 'end' in data_sample.keys():
                end = data_sample['end']
            if 'asr' in data_sample.keys() and self.use_asr:
                asr = data_sample['asr'].lower()
                tag = asr
        return caption, tag, start, end

    def get_caption_and_timeinfo_wrapper(self, img_idx, cap_idx):
        row = self.get_row_from_tsv(self.cap_tsv, img_idx)
        data_sample = json.loads(row[1])
        caption, asr_or_tag, start, end = self.get_caption_and_timeinfo(
            data_sample, cap_idx)
        return caption, asr_or_tag, start, end

    def get_caption_file_in_coco_format(self):
        # for evaluation
        cap_file_coco_format = find_file_path_in_yaml(
            self.cfg.get('caption_coco_format', None), op.join(self.root, 'metadata'))
        if cap_file_coco_format:
            return cap_file_coco_format
        test_split = op.basename(self.yaml_file).split('.')[0]
        return op.join(self.root, 'metadata', test_split + '_caption_coco_format.json')

    def get_captions_by_key(self, key):
        # get a list of captions for image (by key)
        img_idx = self.key2index[key]
        cap_info = json.loads(self.cap_tsv[img_idx][1])
        return [c['caption'] for c in cap_info]

    def get_video_key(self, idx):
        # line_no = self.get_line_no(idx)
        # return self.label_tsv[line_no][0]
        return self.get_row_from_tsv(self.label_tsv, idx)[0]

    def apply_augmentations(self, frames):
        # if failed to decode video, generate fake frames (should be corner case)
        if frames is None:
            frames = np.zeros((self.decoder_num_frames, self.img_res,
                               self.img_res, 3)).astype(np.uint8)
        # (T, C, H, W) -> (T, H, W, C), channel is RGB
        elif 'torch' in str(frames.dtype):
            frames = frames.numpy()
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = frames.astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
        num_of_frames, height, width, channels = frames.shape

        frame_list = []
        for i in range(self.decoder_num_frames):
            if num_of_frames == 1:
                # if it is from image-caption dataset, we duplicate the image
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[0]))
            else:
                # if it is from video-caption dataset, we add each frame to the list
                # convert numpy to PIL format, compatible to augmentation operations
                frame_list.append(Image.fromarray(frames[i]))

        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        crop_frames = self.raw_video_prcoess(frame_list)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames

    def convert_string_to_float_array(self, s):
        # Remove brackets and extra characters
        s = str(s)
        
        s = s.replace('[', '').replace(']', '').replace(',', ' ')
        s = s.replace("'", "").replace('"', '')

        # Split the string into individual number strings
        number_strings = s.split()

        # Convert each number string to a float
        float_numbers = [float(num) for num in number_strings]
        float_tensor = torch.tensor(float_numbers, dtype=torch.float32)

        return float_tensor
        #return np.array(float_numbers, dtype=np.float32)

    def get_image(self, bytestring):
        # output numpy array (T, C, H, W), channel is RGB, T = 1
        #cv2_im = img_from_base64(bytestring)
        cv2_im = self.convert_string_to_float_array(bytestring)
        #cv2_im = cv2_im[:, :, ::-1]  # COLOR_BGR2RGB
        # cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        #output = np.transpose(cv2_im[np.newaxis, ...], (0, 3, 1, 2))
        return cv2_im

    def get_frames_from_tsv(self, binary_frms):
        # get pre-extracted video frames from tsv files
        frames = []
        #_C, _H, _W = 3, 224, 224                               #CHANGE
        if self.decoder_num_frames > len(binary_frms):
            print(
                f"Corrupt videos, requested {self.decoder_num_frames} frames, "
                f"but got only {len(binary_frms)} frames, will return all zeros instead"
            )
            return np.zeros((self.decoder_num_frames, _C, _H, _W),
                            dtype=np.int64)

        def sampling(start, end, n):
            if n == 1:
                return [int(round((start + end) / 2.))]
            if n < 1:
                raise Exception("behaviour not defined for n<2")
            step = (end - start) / float(n - 1)
            return [int(round(start + x * step)) for x in range(n)]

        for i in sampling(0, len(binary_frms) - 1, self.decoder_num_frames):
            try:
                image = self.get_image(binary_frms[i])                                                      #CHANGE
            except Exception as e:
                print(f"Corrupt frame at {i}")
                #image = np.zeros((1, _C, _H, _W), dtype=np.int64)
            #_, _C, _H, _W = image.shape                                                 #CHANGE
            frames.append(image)
        return torch.vstack(frames)

    def decode_and_get_frames(self, clip_path_name, start=None, end=None):
        # online decode raw video file, and get video frames
        # output tensor (T, C, H, W), channel is RGB, T = self.decoder_num_frames
        if 'TVC' in clip_path_name:
            # default clip_path_name: datasets/TVC/videos/{tv_show}/{tv_show}_clips/{tv_show}_{seasoninfo}/{video_id}.mp4_{start_time}_{end_time}
            # To load video file, we will need to remove start&end info here
            resolved_video_path = '_'.join(clip_path_name.split('_')[0:-2])
        else:  # VATEX, MSVD, MSRVTT, Youcook2
            resolved_video_path = clip_path_name
        frames, video_max_pts = extract_frames_from_video_path(
            resolved_video_path, self.decoder_target_fps,
            self.decoder_num_frames, self.decoder_multi_thread_decode,
            self.decoder_sampling_strategy, self.decoder_safeguard_duration,
            start, end)
        return frames

    def get_visual_data(self, idx, start=None, end=None):
        row = self.get_row_from_tsv(self.visual_tsv, idx)
        # if the input is a video tsv with only video file paths, 
        # extract video frames on-the-fly, and return a video-frame tensor
        if row[0] == row[-1]: 
            return self.decode_and_get_frames(row[-1], start, end), True
        # if the input is a video tsv with frames pre-extracted,
        # return a video-frame tensor
        elif len(row) >= self.decoder_num_frames +1:            
            return self.get_frames_from_tsv(row[1:]), True      
        # if the input is a image tsv, return image numpy array
        else: 
            return self.get_image(row[-1]), False

    def __len__(self):
        return len(self.img_line_list)

    def __getitem__(self, idx):
        if self.args.debug_speed:
            idx = idx % self.args.effective_batch_size


        img_idx, cap_idx = self.get_image_cap_index(idx)

        img_key = self.image_keys[img_idx]


        caption_sample, tag, start, end = self.get_caption_and_timeinfo_wrapper(
            img_idx, cap_idx)
        # tag = ' ' start = None end = None is_video = True
        # get image or video frames
        # frames: (T, C, H, W),  is_video: binary tag
        raw_frames, is_video = self.get_visual_data(img_idx, start, end)

        # apply augmentation. frozen-in-time if the input is an image
        # preproc_frames: (T, C, H, W), C = 3, H = W = self.img_res, channel is RGB
        #preproc_frames = self.apply_augmentations(raw_frames)
        preproc_frames = raw_frames

        # tokenize caption and generate attention maps
        # it will consider only # of visual tokens for building attention maps. # is args.max_img_seq_length
        if isinstance(caption_sample, dict):
            caption = caption_sample["caption"]
        else:
            caption = caption_sample
            caption_sample = None


        return img_key, caption, preproc_frames



class VisionLanguageTSVYamlDataset(VisionLanguageTSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """

    def __init__(self,
                 args,
                 yaml_file,
                 tokenizer,
                 tensorizer=None,
                 is_train=None,
                 on_memory=False):
        # print('Init video/image captioning dataloader...')
        super(VisionLanguageTSVYamlDataset,
              self).__init__(args, yaml_file, tokenizer, tensorizer, is_train,
                             on_memory)