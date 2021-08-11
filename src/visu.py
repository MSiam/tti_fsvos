import os
import cv2
from PIL import Image
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib
from typing import List
from src.flow_viz import flow_to_image

cmaps = ['winter', 'hsv', 'Wistia', 'BuGn']

def make_overlay_img(img, mask):
    im= Image.fromarray(np.uint8(img))
    im= im.convert('RGBA')

    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    mask_color[mask==1] = (255,0,0)
    overlay= Image.fromarray(np.uint8(mask_color))
    overlay= overlay.convert('RGBA')

    im= Image.blend(im, overlay, 0.7)
    blended_arr= np.array(im)[:,:,:3]
    img2= img.copy()
    img2[mask==1,:]= blended_arr[mask==1,:]
    return img2[:,:,::-1]

def create_dir(new_dir, method, save_path):
    new_dir = save_path.replace(method, '%s/%s/'%(method, new_dir))
    new_dir = '/'.join(new_dir.split('/')[:-1])
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def crop_invalid_rect(path, to_crop):
    img = cv2.imread(path)
    img_shape = img.shape[:2]

    if to_crop.shape[0] == to_crop.shape[1]:
        scale = to_crop.shape[0] / np.max(img_shape)
        scale = (scale, scale)
    else:
        scale = (to_crop.shape[0]/ img_shape[0], to_crop.shape[1]/img_shape[1])

    start_crop = (int(scale[0] * img_shape[0]), int(scale[1] * img_shape[1]))
    to_crop = to_crop[:start_crop[0], :start_crop[1]]
    return to_crop

def make_episode_visualization_cv2(img_s: np.ndarray,
                                   img_q: np.ndarray,
                                   gt_s: np.ndarray,
                                   gt_q: np.ndarray,
                                   path_s: str,
                                   path_q: str,
                                   preds: np.ndarray,
                                   save_path: str,
                                   flow_q: np.ndarray = None,
                                   mean: List[float] = [0.485, 0.456, 0.406],
                                   std: List[float] = [0.229, 0.224, 0.225]):

    # 0) Preliminary checks
    assert len(img_s.shape) == 4, f"Support shape expected : K x 3 x H x W or K x H x W x 3. Currently: {img_s.shape}"
    assert len(img_q.shape) == 3, f"Query shape expected : 3 x H x W or H x W x 3. Currently: {img_q.shape}"
    assert len(preds.shape) == 4, f"Predictions shape expected : T x num_classes x H x W. Currently: {preds.shape}"
    assert len(gt_s.shape) == 3, f"Support GT shape expected : K x H x W. Currently: {gt_s.shape}"
    assert len(gt_q.shape) == 2, f"Query GT shape expected : H x W. Currently: {gt_q.shape}"
    # assert img_s.shape[-1] == img_q.shape[-1] == 3, "Images need to be in the format H x W x 3"
    if img_s.shape[1] == 3:
        img_s = np.transpose(img_s, (0, 2, 3, 1))
    if img_q.shape[0] == 3:
        img_q = np.transpose(img_q, (1, 2, 0))

    assert img_s.shape[-3:-1] == img_q.shape[-3:-1] == gt_s.shape[-2:] == gt_q.shape

    if img_s.min() <= 0:
        img_s *= std
        img_s += mean

    if img_q.min() <= 0:
        img_q *= std
        img_q += mean

    img_s = np.array(img_s * 255, np.uint8)
    img_q = np.array(img_q * 255, np.uint8)

    T, num_classes, H, W = preds.shape
    K = img_s.shape[0]

    method = save_path.split('/')[-2]
    create_dir('supports', method, save_path)
    create_dir('queries', method, save_path)
    create_dir('preds', method, save_path)

    for i, (path, img, mask) in enumerate(zip(path_s, img_s, gt_s)):
        img = crop_invalid_rect(path, img)
        mask = crop_invalid_rect(path, mask)
        overlay_img = make_overlay_img(img, mask)
        cv2.imwrite(save_path.replace(method, '%s/supports/'%method).replace('.png', '_sprt%d.png'%i), overlay_img)

    img_q = crop_invalid_rect(path_q, img_q)

    preds = np.argmax(preds[0], axis=0)
    preds = crop_invalid_rect(path_q, preds)

    gt_q = crop_invalid_rect(path_q, gt_q)

    overlay_img_q = make_overlay_img(img_q, gt_q)
    cv2.imwrite(save_path.replace(method, '%s/queries/'%method), overlay_img_q)

    overlay_img_pred = make_overlay_img(img_q, preds)
    cv2.imwrite(save_path.replace(method, '%s/preds/'%method), overlay_img_pred)

def make_episode_visualization(img_s: np.ndarray,
                               img_q: np.ndarray,
                               gt_s: np.ndarray,
                               gt_q: np.ndarray,
                               preds: np.ndarray,
                               save_path: str,
                               flow_q: np.ndarray = None,
                               mean: List[float] = [0.485, 0.456, 0.406],
                               std: List[float] = [0.229, 0.224, 0.225]):

    # 0) Preliminary checks
    assert len(img_s.shape) == 4, f"Support shape expected : K x 3 x H x W or K x H x W x 3. Currently: {img_s.shape}"
    assert len(img_q.shape) == 3, f"Query shape expected : 3 x H x W or H x W x 3. Currently: {img_q.shape}"
    assert len(preds.shape) == 4, f"Predictions shape expected : T x num_classes x H x W. Currently: {preds.shape}"
    assert len(gt_s.shape) == 3, f"Support GT shape expected : K x H x W. Currently: {gt_s.shape}"
    assert len(gt_q.shape) == 2, f"Query GT shape expected : H x W. Currently: {gt_q.shape}"
    # assert img_s.shape[-1] == img_q.shape[-1] == 3, "Images need to be in the format H x W x 3"
    if img_s.shape[1] == 3:
        img_s = np.transpose(img_s, (0, 2, 3, 1))
    if img_q.shape[0] == 3:
        img_q = np.transpose(img_q, (1, 2, 0))

    assert img_s.shape[-3:-1] == img_q.shape[-3:-1] == gt_s.shape[-2:] == gt_q.shape

    if img_s.min() <= 0:
        img_s *= std
        img_s += mean

    if img_q.min() <= 0:
        img_q *= std
        img_q += mean

    T, num_classes, H, W = preds.shape
    K = img_s.shape[0]

    # Create Grid
    n_rows = T+1
    n_columns = num_classes + 1
    if flow_q is not None and n_columns < 3:
        n_columns += 1

    fig = plt.figure(1, figsize=(20, 5), dpi=300.)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_rows, n_columns),
                     axes_pad=(0.1, 0.3),
                     direction='row',
                     )

    # 1) visualize the support and query objects with ground-truth
    start = int((num_classes+1) / 2) - int((K+1) / 2)

    for j in range(n_columns):
        ax = grid[j]
        if j == start + K:
            img = img_q
            mask = gt_q
            make_plot(ax, img, mask)
        elif j >= start and j < start + K:
            img = img_s[j - start]
            mask = gt_s[j - start]
            make_plot(ax, img, mask)
        elif flow_q is not None:
            flow_img = flow_to_image(flow_q.transpose(1,2,0))
            mask = np.zeros(flow_img.shape[:2])
            make_plot(ax, flow_img, mask)
        ax.axis('off')

    # 2) Visualize the predictions evolving with time
    img = img_q
    for i in range(1, n_rows):
        for j in range(n_columns):
            ax = grid[n_columns*i + j]
            ax.axis('off')
            if j == 0:
                # Overall prediction
                mask = preds.argmax(1)[i-1]
                mask[gt_q==255] = 0
                make_plot(ax,
                          img,
                          mask,
                          cmap_names=cmaps[:num_classes],
                          classes=range(1, num_classes))
                ax.text(-W // 3, H // 2, fr"$t = {i-1}$", rotation=90,
                        verticalalignment='center', fontsize=14)
            else:
                # Overall prediction
                mask = preds[i-1, j-1]
                make_plot(ax,
                          img,
                          mask)
    fig.tight_layout()
    fig.savefig(save_path)
    fig.clf()

def make_plot(ax: matplotlib.axes.Axes,
              img: np.ndarray,
              mask: np.ndarray,
              cmap_names: List[str] = ['rainbow'],
              classes: List[int] = None):

    ax.imshow(np.array(img * 255, np.uint8))
    if classes:  # For the overall segmentation map
        for class_, cmap_name in zip(classes, cmap_names):
            cmap = eval(f'plt.cm.{cmap_name}')
            new_mask = mask.copy()
            new_mask[mask == class_] = 1
            new_mask[mask != class_] = 0
            alphas = Normalize(0, .3, clip=True)(new_mask)
            alphas = np.clip(alphas, 0., 0.5)  # alpha value clipped at the bottom at .4
            colors = Normalize()(new_mask)
            colors = cmap(colors)
            colors[..., -1] = alphas
            ax.imshow(colors, cmap=cmap)  # interpolation='none'
    else:  # For probability maps
        new_mask = mask.copy()
        new_mask[mask == 255] = 0
        cmap = eval(f'plt.cm.{cmap_names[0]}')
        alphas = Normalize(0, .3, clip=True)(new_mask)
        alphas = np.clip(alphas, 0., 0.5)  # alpha value clipped at the bottom at .4
        colors = Normalize()(new_mask)
        colors = cmap(colors)
        colors[..., -1] = alphas
        ax.imshow(colors, cmap=cmap)  # interpolation='none'
