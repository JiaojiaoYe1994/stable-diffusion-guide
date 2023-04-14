#!/usr/bin/env python
# coding: utf-8
# Author: Jiaojiao Ye

# The Stable Diffusion Guide 

# ## Prompt engineering

# In[2]:


model_id = "runwayml/stable-diffusion-v1-5"    


# ## Speed Optimization

# In[11]:


from diffusers import StableDiffusionPipeline 
import torch

pipe = StableDiffusionPipeline.from_pretrained(model_id)


# In[8]:


prompt = "portrait photo of a old warrior chief"  


# In[12]:


pipe = pipe.to("cuda")  


# In[13]:


generator = torch.Generator("cuda").manual_seed(0)  


# In[14]:


image = pipe(prompt, generator=generator).images[0]                                                                                                                                                                                           
image.save('./outputs/output_fp32.png')     


# In[17]:


import torch                                                                                                                                                                                                                                  

pipe = StableDiffusionPipeline.from_pretrained(model_id,revision="fp16", torch_dtype=torch.float16)                                                                                                                                                           
pipe = pipe.to("cuda") 


# In[33]:


generator = torch.Generator("cuda").manual_seed(0)                                                                                                                                                                                            

with torch.autocast("cuda"):
    image = pipe(prompt, generator=generator).images[0]                                                                                                                                                                                           
image.save('./outputs/output_fp16.png')      


# In[35]:


pipe.scheduler.compatibles    


# In[37]:


from diffusers import DPMSolverMultistepScheduler                                                                                                                                                                                             
                                                                                                                                                                                                                                              
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  


# In[41]:


generator = torch.Generator("cuda").manual_seed(0)                                                                                                                                                                                            

with torch.autocast("cuda"):
    image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]                                                                                                                                                                   
image.save('./outputs/output_fp16_20steps.png')                   


# ## Memory Optimization

# In[42]:


def get_inputs(batch_size=1):                                                                                                                                                                                                                 
  generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]                                                                                                                                                             
  prompts = batch_size * [prompt]                                                                                                                                                                                                             
  num_inference_steps = 20                                                                                                                                                                                                                    

  return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}     


# In[43]:


from PIL import Image                                                                                                                                                                                                                         

def image_grid(imgs, rows=2, cols=2):                                                                                                                                                                                                         
    w, h = imgs[0].size                                                                                                                                                                                                                       
    grid = Image.new('RGB', size=(cols*w, rows*h))                                                                                                                                                                                            
                                                                                                                                                                                                                                              
    for i, img in enumerate(imgs):                                                                                                                                                                                                            
        grid.paste(img, box=(i%cols*w, i//cols*h))                                                                                                                                                                                            
    return grid                                   


# In[45]:


with torch.autocast("cuda"):
    images = pipe(**get_inputs(batch_size=4)).images                                                                                                                                                                                              
image_grid(images)    
image_grid(images).save('./outputs/output_bs4.png')

# In[46]:


pipe.enable_attention_slicing() 


# In[49]:


with torch.autocast('cuda'):
    images = pipe(**get_inputs(batch_size=8)).images                                                                                                                                                                                              
image_grid(images, rows=2, cols=4) 
image_grid(images, rows=2, cols=4).save('./outputs/output_bs8.png')   

# ## Quality Improvements

# In[50]:


from diffusers import AutoencoderKL                                                                                                                                                                                                           

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")  


# In[52]:


pipe.vae = vae          


# In[53]:


with torch.autocast('cuda'):
    images = pipe(**get_inputs(batch_size=8)).images                                                                                                                                                                                              
image_grid(images, rows=2, cols=4)    

# In[54]:


prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"                                                                                                                                                   


# In[55]:


prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"                                                                                                                                         
prompt                                                                                                                                                                                                                                        


# In[57]:


with torch.autocast('cuda'):
    images = pipe(**get_inputs(batch_size=8)).images                                                                                                                                                                                              
image_grid(images, rows=2, cols=4)                                                                                                                                                                                                            


# In[59]:


prompts = [                                                                                                                                                                                                                                   
    "portrait photo of the oldest warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",                                                                                                                                                                                                                                                                   
    "portrait photo of a old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",                                                                                                                                                                                                                                                                        
    "portrait photo of a warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",                                                                                                                                                                                                                                                                            
    "portrait photo of a young warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",                                                                                                                                                                                                                                                                      
]                                                                                                                                                                                                                                             

generator = [torch.Generator("cuda").manual_seed(1) for _ in range(len(prompts))]  # 1 because we want the 2nd image                                                                                                                          

with torch.autocast('cuda'):
    images = pipe(prompt=prompts, generator=generator, num_inference_steps=25).images                                                                                                                                                             
image_grid(images)                                                                                                                                                                                                                         
image_grid(images).save('./outputs/output_bs8_improved.png')
