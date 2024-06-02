## Reproduction
1. Install dependencies - `pip install -r requirements.txt`
2. Place raw data from kaggle (https://www.kaggle.com/code/joonasyoon/wgan-cp-with-celeba-and-lsun-dataset) into folder `data/raw`
3. Run following commands
``` console
python bin/train_unet_diffuser.py --name unet_hf_default --noise-scheduler DDPMScheduler --denoising-steps 1000
python bin/train_unet_diffuser.py --name unet_hf_ddim --noise-scheduler DDIMScheduler --denoising-steps 150
python bin/train_unet_diffuser.py --name unet_hf_euler --noise-scheduler EulerDiscreteScheduler --denoising-steps 30
```
4. Run all cells in notebook `analyze_models.ipynb`
