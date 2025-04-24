from datasets import load_dataset
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from tqdm import tqdm
import io

if __name__ == "__main__":
    # Create destination directory if it doesn't exist
    dest_path = "./face_embeddings"
    os.makedirs(dest_path, exist_ok=True)
    
    # Load LFW dataset
    print("Loading LFW dataset...")
    ds = load_dataset("vilsonrodrigues/lfw")
    print(ds)
    # Initialize face detection and recognition models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize MTCNN for face detection with fixed_image_size
    mtcnn = MTCNN(
        image_size=160, 
        margin=20, 
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], 
        factor=0.709, 
        post_process=True,
        device=device,
        select_largest=True  # Select largest face only
    )
    
    # Initialize InceptionResnetV1 model for face recognition
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Process and save 1000 face embeddings
    counter = 0
    num_embeddings = 1000
    
    print(f"Processing {num_embeddings} face images...")
for idx, item in tqdm(enumerate(ds['train']), total=num_embeddings):
    if counter >= num_embeddings:
        break
        
    try:
        img = item['image'].convert('RGB')
        
        boxes, probs = mtcnn.detect(img)
        
        if boxes is not None and len(boxes) > 0:
            box = boxes[np.argmax(probs)]
            x1, y1, x2, y2 = [int(b) for b in box]
            
            face_img = img.crop((x1, y1, x2, y2))
            face_img = face_img.resize((160, 160))
            
            face_tensor = torch.FloatTensor(np.array(face_img)).permute(2, 0, 1).unsqueeze(0) / 255.0
            face_tensor = face_tensor.to(device)
            
            with torch.no_grad():
                embedding = resnet(face_tensor)
            
            embedding_np = embedding.cpu().numpy().flatten()
            
            file_name = f"{counter:04d}.npy"
            file_path = os.path.join(dest_path, file_name)
            np.save(file_path, embedding_np)
            
            counter += 1
            
            if counter % 100 == 0:
                print(f"Processed {counter} embeddings")
    
    except Exception as e:
        print(f"Error processing image {idx}: {e}")
        continue
    
    print(f"Successfully saved {counter} face embeddings to {dest_path}")