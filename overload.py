import torch
from base_attack import BaseAttack
from transformers import DetrConfig, AutoImageProcessor, DetrForObjectDetection
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from PIL import Image
import requests
import torch.nn.functional as F
import pdb
from model_zoo import load_from_pretrained
from utils import set_all_seeds
set_all_seeds(0)

class Overload(BaseAttack):
    
    def generate_mask(self, x_shape, y_shape): 
        mask_x = 4
        mask_y = 2
        mask = torch.ones(y_shape,x_shape)      
        x_len = int(x_shape / mask_x)
        y_len = int(y_shape / mask_y)
        

        if self.boxes is not None:
            # (top_left_x, top_left_y, bottom_right_x, bottom_right_y) 
            for i in range(len(self.boxes)):
                detection = self.boxes[i]
                center_x, center_y = (detection[0]+detection[2])/2, (detection[1]+detection[3])/2
                region_x = int(center_x / x_len)
                region_y = int(center_y / y_len)
                
                mask[region_y*y_len:(region_y+1)*y_len, region_x*x_len:(region_x+1)*y_len] -= 0.05        
        return mask
    
    def run_attack(self, image, img_id):
        self.init_input(image, img_id)
        self.generate_bx()
            
        for it in range(self.it_num):
            added_img = self.img_tensor + self.bx
            added_img.clamp_(min=0, max=1)

            _ = self.inference(added_img)
            if it == 0:
                self.mask = self.generate_mask(added_img.shape[3], 
                                               added_img.shape[2]).to(self.device)
            self.update_bx()
            self.logger(it)
            self.update_buffer(added_img, it)
            
        self.save_with_thread_pool(max_workers=256)
        self.write_log()
        
        self.clean_flag = True
        torch.cuda.empty_cache()

            
    def update_bx(self):  

        loss2 = 40*torch.norm(self.bx, p=2)
        target = torch.ones_like(self.prob)
        loss3 = F.mse_loss(self.prob, target, reduction='sum')
        loss = loss3+2*(10000-loss2)
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)
        self.bx.data = torch.clamp(-3.5 * self.mask * self.bx.grad+ self.bx.data, min=-0.04, max=0.04)


def single_test(num_q = 1000, 
                img_id = 0, 
                url = None,
                device = None):
    
    if not url:
        url = "https://farm5.staticflickr.com/4116/4827719363_31f75f0c8f_z.jpg"
    
    image = Image.open(requests.get(url, stream=True).raw)

    thres = 0.25
    
    for i in range(3):
        model, image_processor = load_from_pretrained(i, device)
        model.config.num_queries = num_q
        overload = Overload(
            model = model,
            image_processor = image_processor,
            it_num = 100,
            conf_thres = thres,
            target_idx = None,
            output_dir = f"./output_rt_detr/overload_test_{i}",
            device = device
        )

        overload.run_attack(image, img_id)


    
if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    single_test(device=device)
    