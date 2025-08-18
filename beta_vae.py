import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class BetaVAE:
    def __init__(self):
        """Initialize an untrained, frozen Beta-VAE model"""
        
        # This is a placeholder architecture
        class VAEModel(nn.Module):
            def __init__(self, latent_dim=128):
                super(VAEModel, self).__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 32x32
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8x8
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 4x4
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(256 * 4 * 4, 512),
                    nn.ReLU(),
                    nn.Linear(512, latent_dim)
                )
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256 * 4 * 4),
                    nn.ReLU(),
                    nn.Unflatten(1, (256, 4, 4)),
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 64x64
                    nn.Sigmoid()  # Output values between 0 and 1
                )
            
            def encode(self, x):
                """Encode input to latent space"""
                return self.encoder(x)
            
            def decode(self, z):
                """Decode latent vector to image"""
                return self.decoder(z)
        
        # Instantiate the model
        self.model = VAEModel()
        
        # Freeze the model immediately after instantiation
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def encode(self, image):
        """Encode a PIL Image to latent space"""
        # Apply transformations
        img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        # Encode to latent space
        with torch.no_grad():
            latent_vector = self.model.encode(img_tensor)
        
        # Return as numpy array, detached from computation graph
        return latent_vector.squeeze(0).detach().numpy()
    
    def decode(self, latent_vector):
        """Decode a latent vector back to PIL Image"""
        # Convert numpy array to tensor
        if isinstance(latent_vector, np.ndarray):
            latent_tensor = torch.from_numpy(latent_vector).float().unsqueeze(0)
        else:
            latent_tensor = latent_vector.unsqueeze(0) if latent_vector.dim() == 1 else latent_vector
        
        # Decode from latent space
        with torch.no_grad():
            output_tensor = self.model.decode(latent_tensor)
        
        # Convert to PIL Image
        # Denormalize the output
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        output_tensor = output_tensor * std + mean
        
        # Clamp values to [0, 1] and convert to PIL
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_tensor = output_tensor.squeeze(0)  # Remove batch dimension
        
        # Convert to PIL Image
        output_image = transforms.ToPILImage()(output_tensor)
        
        return output_image

if __name__ == "__main__":
    # Demonstration usage
    import numpy as np
    
    # Create a dummy random PIL image
    dummy_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_array)
    
    # Instantiate BetaVAE
    vae = BetaVAE()
    
    # Encode the image
    latent_vector = vae.encode(dummy_image)
    print(f"Latent vector shape: {latent_vector.shape}")
    
    # Decode the vector
    reconstructed_image = vae.decode(latent_vector)
    print(f"Reconstructed image size: {reconstructed_image.size}")
    
    print("Beta-VAE demonstration completed successfully!")
