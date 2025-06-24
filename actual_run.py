import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import sep
import math
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def get_slice(image, theta, order):
    ni = int(order)
    ni2 = ni//2
    ny, nx = image.shape
    x0 = nx//2
    y0 = ny//2
    rad = np.radians(theta)
    ct = np.cos(rad)
    st = np.sin(rad)
    sl = np.zeros(nx, dtype=complex)
    x = np.arange(nx)
    y = np.arange(ny)
    if(theta <-45 or theta >45):
        Xy = x0 + (y-y0)*ct/st
        xmin = (Xy - ni2).astype(int)
        for i in range(ni):
            Xv = np.clip(xmin+i,0,nx-1)
            sl += image[y,Xv]*np.sinc(Xy-Xv)
    else:
        Yx = y0 + (x-x0)*st/ct
        ymin = (Yx - ni2).astype(int)
        for i in range(ni):
            Yv = np.clip(ymin+i, 0, ny-1)
            sl += image[Yv,x]*np.sinc(Yx-Yv)
    
    return sl


def fast_radon(img, order):
    ig = img.copy()
    theta = np.arange(-180,180,0.5)
    ny,nx = img.shape
    if ny != nx:
        return 0, 0
    nt = len(theta)
    rt = np.zeros((nx,nt))
    fimg = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ig)))
    for i in range(len(theta)):
        sl = get_slice(fimg, theta[i], order)
        rt[:,i] = np.real(np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(sl))))
        if theta[i] <-45:
            rt[:,i] = rt[::-1,i]
    max_index = np.unravel_index(np.argmax(rt, axis=None), rt.shape)
    row = max_index[0]  # Row index of the maximum value
    col = max_index[1]
    return row, col

def generate_filters(imsize):
    l = 100
    filters = []

    while(l<imsize):
        arr = np.ones(l)
        narr = arr/l
        filters.append({
            "length" : l,
            "filter" : arr
        })
        l+=1
    return filters

def get_line_profile(x,y,q,ig,normalise = False):
    x = np.array(x)
    y = np.array(y)
    x1 = x[0]
    y1 = y[0]
    x2 = x[-1]
    y2 = y[-1]
    ny,nx = ig.shape

    if x2 == x1:
        original_slope = float('inf')
    else :
        original_slope = (y2-y1)/(x2-x1)
    
    if original_slope == 0:
        orth_slope = float('inf')
    elif original_slope == float('inf'):
        orth_slope = 0
    else:
        orth_slope = -1/original_slope

    if orth_slope == float('inf'):
        xstart, ystart = x, y- (q // 2)
        xend, yend = x, y+ (q //2)
    elif orth_slope == 0:
        xstart, ystart = x-(q//2) , y
        xend, yend = x+(q//2), y
    else:
        dx = int((q/2)/(1+orth_slope**2)**0.5)
        dy = int(dx*orth_slope)
        xstart,ystart = x-dx, y-dy
        xend, yend = x+dx, y+dy
    
    xpoints = []
    ypoints = []
    mint = []

    for xs, ys, xe, ye,xi,yi in zip(xstart, ystart, xend, yend,x,y):
        if xs >= nx or ys >= ny or xe >= nx or ye >= ny or xs < 0 or ys < 0 or xe < 0 or ye < 0:
            continue
        xpoints.append(xi)
        ypoints.append(yi)
        dbx = abs(xs-xe)
        dby = abs(ys-ye)
        sx = 1 if xs < xe else -1
        sy = 1 if ys < ye else -1
        err = dbx - dby

        inten = []
        while True:
            inten.append(ig[ys,xs])
            if xs == xe and ys == ye:
                break
            e2 = 2*err
            if e2 > -dby:
                err -= dby
                xs += sx
            if e2 < dbx:
                err += dbx
                ys += sy
        mint.append(np.mean(inten))
    mint = np.array(mint)
    
    if normalise:
        arm, arstd = np.mean(mint), np.std(mint)
        arnorm = (mint-arm)/arstd
    
        return np.array(xpoints), np.array(ypoints), np.array(arnorm)
    return np.array(xpoints), np.array(ypoints), np.array(mint)


def line(theta,rho,ig,flag=0,low_snr=False):
    ny,nx = ig.shape
    x0 = nx//2
    y0 = ny//2
    xcord = []
    ycord = []
    vals = []
    if(theta <-45 or theta >45):
        for x in range(nx):
            y = round(rho-((x-x0)*np.cos(np.radians(theta))/np.sin(np.radians(theta))) + y0)
            if(y-y0 in range(ny)):
                xcord.append(x)
                ycord.append(y-y0)
                vals.append(ig[y-y0,x])
    else:
        for y in range(ny):
            x = round(rho-((y-y0)*np.sin(np.radians(theta))/np.cos(np.radians(theta))) + x0)
            if(x-x0 in range(nx)):
                xcord.append(x-x0)
                ycord.append(y)
                vals.append(ig[y,x-x0])
    
    if len(xcord) == 0:
        return [],[]
    

    _, _, mprofile = get_line_profile(xcord,ycord,8,ig)

    if len(mprofile) == 0:
        return [], []
    
    filters = generate_filters(len(xcord))
    ls = []
    inds = []
    peaks = []
    thresh = 20000 if low_snr else 120000

    for fltr in filters:
        match_filter = fltr['filter']
        if len(mprofile) > 0 and len(match_filter)>0:
            match_filtered_data = np.convolve(mprofile, match_filter, mode='same')
            ind = np.argmax(match_filtered_data)
            peak = match_filtered_data[ind]
            ls.append(fltr['length'])
            inds.append(ind)
            peaks.append(peak)

    max_ind = np.argmax(peaks)
    mpeak = peaks[max_ind]
    ml = ls[max_ind]
    mind = inds[max_ind]

    newx, newy = [], []
    for i in range((mind-ml//2),(mind+ml//2)):
        if i in range(len(xcord)):
            newx.append(xcord[i])
            newy.append(ycord[i])

    if((len(newx) == 0 or mpeak < thresh) and flag == 0):
        return line(theta,ig.shape[0]-1-rho,ig,1,low_snr=low_snr)
    elif((len(newx) == 0 or mpeak < thresh) and flag == 1):
        return [],[]
    else :
        return np.array(newx), np.array(newy)
    
def read_fits_file(path):
    hdul = fits.open(path)
    headers =  hdul[0].header
    image = hdul[0].data
    hdul.close()
    return headers, image

def get_scaled_image(image,scale=255):
    z = ZScaleInterval()
    z1, z2 = z.get_limits(image)
    vmax = z2
    vmin = z1
    image[image > vmax] = vmax
    image[image < vmin] = vmin
    image = (image - vmin)/(vmax -vmin)
    image = (scale*image).astype(np.uint8)
    return image

def get_source_extracted_image(image, low_snr = False):
    bkg = sep.Background(image.byteswap().newbyteorder(),bw=20, bh=20, fw=10, fh=10) 
    bkg_image = bkg.back(dtype=int)
    image_subtracted = image - bkg_image

    thresh = 50 if low_snr else 200

    objects = []
    try:
        objects = sep.extract(image_subtracted, thresh)
    except:
        print(f"Error at Source Extraction: Failed to get sources at thresh {thresh} attempting thresh {thresh+5}")
        objects = []
        thresh += 5
    
    mask = np.zeros(image_subtracted.shape, dtype = np.uint8)
    for obj in objects:
        x = obj['x']
        y = obj['y']
        a = obj['a']

        theta = np.radians(obj['theta'])

        sep.mask_ellipse(mask,x,y,6*a,6*a,theta)
    data = np.where(mask == 1, 0, image_subtracted)

    return data

def detect_line(image, low_snr = False):
    image_d = image.copy()
    scaled_image = get_scaled_image(image_d)
    data = get_source_extracted_image(image=scaled_image, low_snr=low_snr)

    start = 0
    jump = data.shape[1]
    limit =  data.shape[0]/data.shape[1]
    xfinal = []
    yfinal = []
    shifter = 100

    plt.figure(figsize=(100, 80))
    plt.imshow(image, cmap='gray',vmin=0, vmax=255)

    while start < limit:
        img = data[jump*start:jump*(start+1),:]
        rho, theta = fast_radon(img, 7)

        tt = math.floor(np.abs((theta*5/10)-180))
        if tt in range(0,3) or tt in range(87,93) or tt in range(177,183):
            start += 1
            continue

        xx, yy = line((theta*5/10)-180, rho, img, low_snr=low_snr)

        if len(xx) > 0:
            yy += start*jump
            st = jump*start
            end = jump*(start+1)
            prev_x = xx.tolist().copy()
            prev_y = yy.tolist().copy()

            while True:
                st -= shifter
                end -= shifter

                if st <= 0:
                    break
                else:
                    trho, ttheta = fast_radon(data[st:end, :], 7)
                    ttt = math.floor(np.abs((theta*5/10)-180))
                    if ttt in range(0,3) or tt in range(87,93):
                        break
                    else:
                        txx, tyy = line((ttheta*5/10)-180,trho,data[st:end,:], low_snr=low_snr)
                        if len(txx) > 0:
                            tyy += st
                            prev_x.extend(txx.tolist())
                            prev_y.extend(tyy.tolist())
                            tpoints = list(set(zip(prev_x,prev_y)))
                            xt, yt = zip(*tpoints)
                            prev_x = list(xt)
                            prev_y = list(yt)
                        else:
                            break

            st = jump*start
            end = jump*(start+1)

            while True:
                st += shifter
                end += shifter

                if end >= data.shape[0]:
                    break
                else:
                    trho, ttheta = fast_radon(data[st:end, :], 7)
                    ttt = math.floor(np.abs((theta*5/10)-180))
                    if ttt in range(0,3) or tt in range(87,93):
                        break
                    else:
                        txx, tyy = line((ttheta*5/10)-180,trho,data[st:end,:], low_snr=low_snr)
                        if len(txx) > 0:
                            tyy += st
                            prev_x.extend(txx.tolist())
                            prev_y.extend(tyy.tolist())
                            tpoints = list(set(zip(prev_x,prev_y)))
                            xt, yt = zip(*tpoints)
                            prev_x = list(xt)
                            prev_y = list(yt)
                        else:
                            break

            if len(prev_x) > len(xx):
                try:
                    points = np.column_stack((prev_x, prev_y))
                    distance_threshold = 1 #Set based on data scale
                    min_neighbors = 3

                    kdtree = KDTree(points)

                    neighbors_count = np.array([
                        len(kdtree.query_ball_point(point, distance_threshold)) for point in points
                    ])

                    dense_mask = neighbors_count > min_neighbors
                    dense_points = points[dense_mask]

                    x_dense, y_dense = dense_points[:, 0], dense_points[:, 1]
                except:
                    x_dense, y_dense = [], []

                if len(x_dense) > 0:
                    print("x dense greater than 0")
                    xx = np.array(x_dense)
                    yy = np.array(y_dense)
                else:
                    xx = np.array(prev_x)
                    yy = np.array(prev_y)
                
                f = 0
                while st > jump*(start+1):
                    start += 1
                    f = 1
                if f == 1:
                    start -= 1


            plt.scatter(xx, yy)
            
        start += 1
    plt.show()




path = "E:\\wfh\\streak_obj_association\\DATA\\images\\20230513\\20230513_r_15h34m.fits"
headers, image = read_fits_file(path)

scaled_image = get_scaled_image(image.copy())
detect_line(scaled_image, low_snr=False)