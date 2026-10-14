"""Generate a tiny synthetic cine for the existing inhouse_cine reader."""
from pathlib import Path
import numpy as np

def main():
    root = Path('outputs/toy_data')
    root.mkdir(parents=True, exist_ok=True)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64), indexing='ij')
    image = np.zeros((2, 24, 64, 64, 1), dtype=np.complex64)
    for s in range(2):
        for t in range(24):
            radius = np.sqrt((xx + 0.07*s)**2 + yy**2)
            wall_radius = 0.30 + 0.035*np.cos(2*np.pi*t/24)
            amplitude = 0.2*np.exp(-2.4*(xx**2+yy**2)) + 0.8*np.exp(-((radius-wall_radius)/0.035)**2)
            image[s,t,:,:,0] = amplitude*np.exp(1j*(0.4*xx + 0.2*yy + 0.05*np.sin(2*np.pi*t/24)))
    maps = np.empty((2, 1, 64, 64, 4), dtype=np.complex64)
    for c in range(4):
        angle = c*np.pi/2
        maps[:,:,:,:,c] = (1 + 0.25*(xx*np.cos(angle)+yy*np.sin(angle)))*np.exp(1j*(xx*np.sin(angle)-yy*np.cos(angle)))
    maps /= np.sqrt(np.sum(np.abs(maps)**2, axis=-1, keepdims=True))
    np.save(root/'norm_img_ToyCine.txt.npy', image)
    np.save(root/'cc_smap_4_ToyCine.txt.npy', maps)
    (root/'manifest.csv').write_text('filename,split,cohort_type\nToyCine,development,synthetic\n', encoding='utf-8')
    print(f'Synthetic cine written to {root}. No patient data or pretrained weights are used.')

if __name__ == '__main__':
    main()
