import torch
from base_attack import BaseAttack
from PIL import Image
import requests
import torch.nn.functional as F
from model_zoo import load_from_pretrained
from torch.optim import Adam
from utils import set_all_seeds
set_all_seeds(0)

class SlowTrack(BaseAttack):
    
    def allocation_strategy(self, frameRate):
        K = 1000
        next_reactivate = {}  
        max_active = 0  
        strategy = []
        tracker_id = 0
        t = 0
        while(t<K):
            if t == 0:
                strategy.append(tracker_id)
                next_reactivate[tracker_id] = t+frameRate+1
                tracker_id += 1
                t+=1
            else:
                reactivate_time = min(next_reactivate.values())
                reactivate_tracker, reactivate_time = min(next_reactivate.items(), key=lambda x: x[1])
                if reactivate_time - t <= 1:
                    strategy.append(reactivate_tracker)
                    next_reactivate[reactivate_tracker] = t+frameRate+1
                    t+=1
                else:
                    strategy.append(tracker_id)
                    strategy.append(tracker_id)
                    next_reactivate[tracker_id] = t+frameRate+2
                    tracker_id += 1
                    t+=2
        return strategy, tracker_id
    
    def update_bx(self, strategy, max_tracker_num):

        per_num_b = (25*45)/max_tracker_num
        per_num_m = (50*90)/max_tracker_num
        per_num_s = (100*180)/max_tracker_num

        sel_scores_b = self.prob[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
        sel_scores_m = self.prob[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
        sel_scores_s = self.prob[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]

        sel_dets = torch.cat((sel_scores_b, sel_scores_m, sel_scores_s), dim=0)
        targets = torch.ones_like(sel_dets)
        loss1 = 10*(F.mse_loss(sel_dets, targets, reduction='sum'))
        loss2 = 40*torch.norm(self.bx, p=2) 
        targets = torch.ones_like(self.prob) 
        loss3 = 1.0*(F.mse_loss(self.prob, targets, reduction='sum') )
        loss = loss1+loss2+loss3
        
        loss.requires_grad_(True)
        self.adam_opt.zero_grad()
        loss.backward(retain_graph=True)
        
        self.bx.grad = self.bx.grad / (torch.norm(self.bx.grad,p=2) + 1e-20)
        self.bx.data = torch.clamp(-1.5 * self.bx.grad + self.bx.data, min=-0.04, max=0.04)
        
        return self.bx
    
    def run_attack(self, image, img_id):
        max_tracker_num = int(6)

        self.init_input(image, img_id)
        self.generate_bx()
        self.adam_opt = Adam([self.bx], lr=1e-3, amsgrad=True)
        strategy, max_tracker_num = self.allocation_strategy(max_tracker_num)
        
        for it in range(self.it_num):
            added_img = self.img_tensor + self.bx
            added_img.clamp_(min=0, max=1)
            _ = self.inference(added_img)
            self.bx = self.update_bx(strategy[it], max_tracker_num)
            # import pdb; pdb.set_trace()
            self.logger(it)
            self.update_buffer(added_img, it)
            
        self.save_with_thread_pool(max_workers=256)
        self.write_log()
        
        self.clean_flag = True
        torch.cuda.empty_cache()

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
        slowtrack = SlowTrack(
            model = model,
            image_processor = image_processor,
            it_num = 100,
            conf_thres = thres,
            target_idx = None,
            output_dir = f"./output_rt_detr/slow_test_{i}",
            device = device
        )
        

        slowtrack.run_attack(image, img_id)


    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    single_test(device=device)