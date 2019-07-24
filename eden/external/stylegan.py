import os
import sys
import pickle
import numpy as np
import scipy
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import tensorflow as tf
import random
import math
import cv2
import eden.setup
import eden.utils
from eden.utils import processing
from eden.utils.utils import check_if_url


tflib_init = False


def setup(checkpoint):
    global tflib_init
    global stylegan, stylegan, dnnlib, tflib, config, Generator, PerceptualModel, load_images, load_model, split_to_batches
    global Gs, fmt

    stylegan = eden.setup.get_external_repo_dir('stylegan')
    sys.path.insert(0, stylegan)
    
    import dnnlib
    import dnnlib.tflib as tflib
    import config
    from encoder.generator_model import Generator
    from encoder.perceptual_model import PerceptualModel, load_images
    from keras.models import load_model
    from encode_images import split_to_batches
    
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    if not tflib_init:
        tflib.init_tf()
        tflib_init = True
    
    is_url = check_if_url(checkpoint)

    if is_url:
        with dnnlib.util.open_url(checkpoint, cache_dir=config.cache_dir) as f:
            G, D, Gs = pickle.load(f)
        print("downloaded network from %s" % checkpoint)
    else:
        checkpoint_path = os.path.join(os.path.join(stylegan, 'checkpoints'), checkpoint)
        with open(checkpoint_path, 'rb') as file:
            G, D, Gs = pickle.load(file)
        print("loaded network %s" % checkpoint_path)


def get_dlatents(latents, labels=None):
    if len(latents.shape)==1:
        latents = latents.reshape((1, 512))
    dlatents = Gs.components.mapping.run(latents, labels) 
    return dlatents
    
        
def load_latents(npy_files):
    if not isinstance(npy_files, list):
        npy_files = [npy_files]
    latents = np.array([np.load(npy_file) for npy_file in npy_files])
    return latents


def generate(latents, labels=None, truncation=1.0, show=False, num_cols=6):
    latents = np.array(latents)
    n = latents.shape[0]
    nr, nc = math.ceil(n / num_cols), num_cols
    for r in range(nr):
        min_idx, max_idx = num_cols * r, min(n, num_cols * (r+1))
        sub_latents = latents[min_idx : max_idx]
        sub_labels = labels[min_idx : max_idx] if labels is not None else None
        if len(sub_latents.shape) < 3:
            sub_latents = get_dlatents(sub_latents, sub_labels)
        images = Gs.components.synthesis.run(sub_latents, labels=sub_labels, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
        images = np.clip(images, 0, 255).astype(np.uint8)
        if show:
            images_ct = np.concatenate([image for image in images], axis=1)
            processing.show(images_ct)
        all_images = np.concatenate([all_images, images]) if r>0 else images
    return all_images


def random_sample(num_images, truncation=1.0, seed=None, show=False, num_cols=6):
    seed = seed if seed is not None else 1000 * random.random()
    latents = np.random.RandomState(int(seed)).randn(num_images, *Gs.input_shapes[0][1:])
    labels = None  # np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
    if show:
        images = generate(latents, labels, truncation, show, num_cols)
    return latents


def random_gaussian_sequence(num_frames, smoothing_frames, random_seed=None):
    shape = [num_frames, np.prod([1, 1])] + Gs.input_shape[1:] # [frame, image, channel, component]
    if random_seed is not None:
        latents = np.random.RandomState(random_seed).randn(*shape).astype(np.float32)
    else:
        latents = np.random.randn(*shape).astype(np.float32)
    latents = scipy.ndimage.gaussian_filter(latents, [smoothing_frames] + [0] * len(Gs.input_shape), mode='wrap')
    latents /= np.sqrt(np.mean(np.square(latents)))
    latents = latents.reshape((num_frames, 512))
    return latents


def get_latent_interpolation(endpoints, frames_to_endpoint, frames_at_endpoint=0, wrap=False, mode='normal', shuffle=False):
    nft, nfa = frames_to_endpoint, frames_at_endpoint
    if shuffle:
        random.shuffle(endpoints)
    num_endpoints = len(endpoints) if isinstance(endpoints, list) else endpoints.shape[0]
    endpoints = np.array(endpoints)
    if endpoints.shape[-2] == 1:   # this is a bit hacky...
        endpoints = endpoints.reshape((num_endpoints, 512))
    num_segments = num_endpoints if wrap else num_endpoints-1
    num_frames = (nft + nfa) * num_segments
    latents = np.zeros((num_frames, 18, 512)) if len(endpoints.shape)==3 else np.zeros((num_frames, 512))
    for e in range(num_segments):
        e1, e2 = e, (e+1)%num_endpoints
        for t in range(nft):
            frame = e1 * (nft + nfa) + t
            r = 0.5 - 0.5 * np.cos(np.pi*t/(nft-1)) if mode == 'ease' else float(t) / nft
            latents[frame] = (1.0 - r) * endpoints[e1] + r * endpoints[e2]
        latents[e1 * (nft + nfa) + nft : (e1 + 1) * (nft + nfa)] = endpoints[e2]
    return latents


def get_latent_interpolation_bspline(endpoints, nf, k, s, shuffle):
    if shuffle:
        random.shuffle(endpoints)
    x = np.array(endpoints)
    x = np.append(x, x[0,:].reshape(1, x.shape[1]), axis=0)
    nd = x.shape[1]
    latents = np.zeros((nd, nf))
    nss = list(range(1, 10)) + [10]*(nd-19) + list(range(10,0,-1))
    for i in tqdm(range(nd-9)):
        idx = list(range(i,i+10))
        tck, u = interpolate.splprep([x[:,j] for j in range(i,i+10)], k=k, s=s)
        out = interpolate.splev(np.linspace(0, 1, num=nf, endpoint=True), tck)
        latents[i:i+10,:] += np.array(out)
    latents = latents / np.array(nss).reshape((512,1))
    return latents.T











        
def generate_movie(latents, labels, out_dir, out_name, truncation=0.5, batch_size=16):
    temp_dir = 'frames%06d'%int(1000000*random.random())
    os.system('mkdir %s'%temp_dir)
    num_frames = latents.shape[0]
    num_batches = int(np.ceil(num_frames/batch_size))
    frame = 1
    for b in tqdm(range(num_batches), desc="Generating %d images in %d batches"%(num_frames, num_batches)):
        min_idx, max_idx = b * batch_size, min((b+1) * batch_size, num_frames-1)
        sub_latents = latents[min_idx:max_idx]
        sub_labels = labels[min_idx : max_idx] if labels is not None else None
        if len(sub_latents.shape) < 3:
            sub_latents = get_dlatents(sub_latents, sub_labels)
        new_images = Gs.components.synthesis.run(sub_latents, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
        for img in new_images:
            Image.fromarray(img, 'RGB').save('%s/frame%05d.png' % (temp_dir, frame))
            frame += 1
    cmd = 'ffmpeg -i %s/frame%%05d.png -c:v libx264 -pix_fmt yuv420p %s/%s.mp4' % (temp_dir, out_dir, out_name)
    os.system(cmd)
    os.system('rm -rf %s'%temp_dir)
    


    
    
    
    
    
    

# encode (specific to checkpoint, path to image, or image object, num_iters etc, record vid?)
def encode(resnet, learning_rate = 0.02, iterations = 200):

    args = eden.utils.DictMap()
    args_other = eden.utils.DictMap()

    args.src_dir = os.path.join(stylegan, 'aligned_images')
    args.generated_images_dir = os.path.join(stylegan, 'generated_images')
    args.dlatent_dir = os.path.join(stylegan, 'latent_representations')
    
    args.load_last = None
    args.dlatent_avg = None
    
    args.model_res = 1024
    args.batch_size = 1

    # Perceptual model params
    args.image_size = 256
    args.resnet_image_size = 256
    args.lr = learning_rate
    args.decay_rate = 0.9
    args.iterations = iterations
    args.decay_steps = 10
    args.load_effnet = None
    args.load_resnet = os.path.join(stylegan, resnet)

    # Loss function options
    args.use_vgg_loss = 0.4
    args.use_vgg_layer = 9
    args.use_pixel_loss = 1.5
    args.use_mssim_loss = 100
    args.use_lpips_loss = 100
    args.use_l1_penalty = 1

    # Generator params
    args.randomize_noise = False
    args.tile_dlatents = False
    args.clipping_threshold = 2.0

    # Masking params
    args.load_mask = False
    args.face_mask = False
    args.use_grabcut = True
    args.scale_mask = 1.5

    # Video params
    args.video_dir = os.path.join(stylegan, 'videos')
    args.output_video = True
    args.video_codec = 'MJPG'
    args.video_frame_rate = 30
    args.video_size = 1024
    args.video_skip = 1

    args.decay_steps *= 0.01 * args.iterations # Calculate steps as a percent of total iterations

    if args.output_video:
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=args.batch_size)

    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    eden.utils.try_make_folder(args.generated_images_dir)
    eden.utils.try_make_folder(args.dlatent_dir)
    eden.utils.try_make_folder(args.video_dir)

    # Initialize generator and perceptual model
    generator = Generator(Gs, args.batch_size, 
                          clipping_threshold=args.clipping_threshold, 
                          tiled_dlatent=args.tile_dlatents, 
                          model_res=args.model_res, 
                          randomize_noise=args.randomize_noise)
    if (args.dlatent_avg is not None):
        generator.set_dlatent_avg(np.load(args.dlatent_avg))

    perc_model = None
    if (args.use_lpips_loss > 0.00000001):
        cache_dir = os.path.join(stylegan, config.cache_dir)
        with dnnlib.util.open_url('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2', cache_dir=cache_dir) as f:
            perc_model = pickle.load(f)
    perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator)

    ff_model = None

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images)//args.batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
        if args.output_video:
            video_out = {}
            for name in names:
                video_out[name] = cv2.VideoWriter(os.path.join(args.video_dir, f'{name}.avi'),
                                                  cv2.VideoWriter_fourcc(*args.video_codec), 
                                                  args.video_frame_rate, 
                                                  (args.video_size,args.video_size))
        perceptual_model.set_reference_images(images_batch)
        dlatents = None
        if (args.load_last is not None): # load previous dlatents for initialization
            for name in names:
                dl = np.expand_dims(np.load(os.path.join(args.load_last, f'{name}.npy')),axis=0)
                if (dlatents is None):
                    dlatents = dl
                else:
                    dlatents = np.vstack((dlatents,dl))
        else:
            if (ff_model is None):
                if os.path.exists(args.load_resnet):
                    print("Loading ResNet Model:")
                    ff_model = load_model(args.load_resnet)
                    from keras.applications.resnet50 import preprocess_input
            if (ff_model is None):
                if os.path.exists(args.load_effnet):
                    import efficientnet
                    print("Loading EfficientNet Model:")
                    ff_model = load_model(args.load_effnet)
                    from efficientnet import preprocess_input
            if (ff_model is not None): # predict initial dlatents with ResNet model
                dlatents = ff_model.predict(preprocess_input(load_images(images_batch,image_size=args.resnet_image_size)))
        if dlatents is not None:
            generator.set_dlatents(dlatents)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations)
        pbar = tqdm(op, leave=False, total=args.iterations)
        vid_count = 0
        best_loss = None
        best_dlatent = None
        for loss_dict in pbar:
            pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v)
                    for k, v in loss_dict.items()]))
            if best_loss is None or loss_dict["loss"] < best_loss:
                best_loss = loss_dict["loss"]
                best_dlatent = generator.get_dlatents()
            if args.output_video and (vid_count % args.video_skip == 0):
                batch_frames = generator.generate_images()
                for i, name in enumerate(names):
                    video_frame = Image.fromarray(batch_frames[i], 'RGB').resize((args.video_size,args.video_size), Image.LANCZOS)
                    video_out[name].write(cv2.cvtColor(np.array(video_frame).astype('uint8'), cv2.COLOR_RGB2BGR))
            generator.stochastic_clip_dlatents()
        print(" ".join(names), " Loss {:.4f}".format(best_loss))

        if args.output_video:
            for name in names:
                video_out[name].release()

        # Generate images from found dlatents and save them
        generator.set_dlatents(best_dlatent)
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()
        
        