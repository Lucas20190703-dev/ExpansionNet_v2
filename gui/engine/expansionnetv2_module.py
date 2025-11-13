
import torch
import pickle
import sys
from argparse import Namespace
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))


from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description

modelWrapper = None

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.coco_tokens = None
        self.sos_idx = None
        self.eos_idx = None
        self.token_path = "./demo_material/demo_coco_tokens.pickle"
        self.image_size = 384
        self.model_dim = 512
        self.N_enc = 3
        self.N_dec = 3
        self.max_seq_len = 74
        self.device = "cpu"
        self.model_path = str(root_path / "github_ignore_material/raw_data/rf_model.pth")
        self.beam_size = 5

        self.drop_args = Namespace(enc=0.0,
                            dec=0.0,
                            enc_input=0.0,
                            dec_input=0.0,
                            other=0.0)
        
        self.coco_tokens = None
        self.sos_idx = None
        self.eos_idx = None

    def load_dictionary(self):
        try:
            with open(self.token_path, 'rb') as f:
                self.coco_tokens = pickle.load(f)
                self.sos_idx = self.coco_tokens['word2idx_dict'][self.coco_tokens['sos_str']]
                self.eos_idx = self.coco_tokens['word2idx_dict'][self.coco_tokens['eos_str']]
        except Exception as e:
            print(f"Error loading dicationary: {e}")

    def load_model(self):
        try:
            self.model = End_ExpansionNet_v2(
                swin_img_size=self.image_size, 
                swin_patch_size=4, 
                swin_in_chans=3,
                swin_embed_dim=192, 
                swin_depths=[2, 2, 18, 2], 
                swin_num_heads=[6, 12, 24, 48],
                swin_window_size=12, 
                swin_mlp_ratio=4., 
                swin_qkv_bias=True, 
                swin_qk_scale=None,
                swin_drop_rate=0.0, 
                swin_attn_drop_rate=0.0, 
                swin_drop_path_rate=0.0,
                swin_norm_layer=torch.nn.LayerNorm, 
                swin_ape=False, 
                swin_patch_norm=True,
                swin_use_checkpoint=False,
                final_swin_dim=1536,

                d_model= self.model_dim, 
                N_enc= self.N_enc,
                N_dec= self.N_dec, 
                num_heads=8, 
                ff=2048,
                num_exp_enc_list=[32, 64, 128, 256, 512],
                num_exp_dec=16,
                output_word2idx=self.coco_tokens['word2idx_dict'],
                output_idx2word=self.coco_tokens['idx2word_list'],
                max_seq_len=self.max_seq_len, 
                drop_args=self.drop_args,
                rank=self.device
            )

            checkpoint = torch.load(self.model_path, map_location=torch.device(self.device))
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        except Exception as e:
            print(f"Error loading model: {e}")
        return None

    def get_caption(self, image_file) -> list:
        image = preprocess_image(image_file, self.image_size)
        beam_search_kwargs = {
            'beam_size': self.beam_size,
            'beam_max_seq_len': self.max_seq_len,
            'sample_or_max': 'max',
            'how_many_outputs': 3,
            'sos_idx': self.sos_idx,
            'eos_idx': self.eos_idx
        }
        
        with torch.no_grad():
            pred, _ = self.model(enc_x=image, enc_x_num_pads=[0], mode='beam_search', **beam_search_kwargs)
        
        predicts = []
        
        for i in range(len(pred[0])):
            p = tokens2description(pred[0][i], self.coco_tokens['idx2word_list'], self.sos_idx, self.eos_idx)
            predicts.append(p)
        
        return predicts
    
def init_expansionnetv2_model():
    global modelWrapper
    modelWrapper =  ModelWrapper()
    modelWrapper.load_dictionary()
    modelWrapper.load_model()

def get_caption(image_file):
    return image_file, modelWrapper.get_caption(image_file)
