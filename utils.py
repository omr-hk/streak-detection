import numpy as np
from astropy.io import fits
from datetime import datetime as dt
import datetime
from astropy.wcs import WCS
from astropy import wcs as wcs_lib
from astropy.visualization import ZScaleInterval
import sep
import math
import os
import glob
import pandas as pd
import gc
import matplotlib.pyplot as plt
from matplotlib import rcParams
# from scipy.spatial import distance_matrix
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import time

rcParams['figure.figsize'] = [100., 80.]

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


def line(theta,rho,ig,flag=0,low_snr=False,debug=False):
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
    #    print("Mprofile empty")
       return [],[]
    
    # fs = "40"
    # ts = 20
    # plt.plot(vals)
    # plt.title("INTENSITY PROFILE", fontsize=fs)
    # plt.ylabel("INTENSITY", fontsize=fs)
    # plt.xlabel("POINTS ON THE LINE", fontsize=fs)
    # plt.xticks(fontsize=ts)
    # plt.yticks(fontsize=ts)
    # plt.show()

    # plt.plot(mprofile)
    # plt.title("MEAN PROFILE", fontsize=fs)
    # plt.ylabel("MEAN", fontsize=fs)
    # plt.xlabel("POINTS ON THE LINE", fontsize=fs)
    # plt.xticks(fontsize=ts)
    # plt.yticks(fontsize=ts)
    # plt.show()
    


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
            if len(peaks) > 50:
                if peaks[-50] == peak:
                    break
            ls.append(fltr['length'])
            inds.append(ind)
            peaks.append(peak)
    
    try:
        fd = vals
        max_ind = np.argmax(peaks)
        # fd = np.abs(np.diff(peaks))
        # wr = np.where(fd <= 0.0005)
        # if len(wr) > 0:
        #     max_ind = wr[0][0]
        # else:
        #     max_ind = np.argmax(peaks)
        mpeak = peaks[max_ind]
        # print("Peak selected is: ",mpeak," thresh: ",thresh," Streak: ",thresh <= mpeak)
        ml = ls[max_ind]
        mind = inds[max_ind]
    except:
        return [],[]
    
    newx, newy = [], []
    for i in range((mind-ml//2),(mind+ml//2)):
        if i in range(len(xcord)):
            newx.append(xcord[i])
            newy.append(ycord[i])
    # for i in range((mind),(mind+ml)):
    #     if i in range(len(xcord)):
    #         newx.append(xcord[i])
    #         newy.append(ycord[i])
            
    
    if debug:
        if((len(newx) == 0) and flag == 0):
            return line(theta,ig.shape[0]-1-rho,ig,1,low_snr=low_snr,debug=debug)
        elif((len(newx) == 0) and flag == 1):
            return [],[]
        else :
            plt.plot(ls,peaks)
            plt.plot(ls[np.argmax(peaks)],peaks[np.argmax(peaks)],'x')
            plt.xlabel("Length of filter", fontsize="30")
            plt.ylabel("Max peak of filter", fontsize="30")
            plt.title("Comparison of filter peaks by length", fontsize=fs)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()
                    
            # plt.plot(fd)
            # plt.title("Intensities of detected line")
            # plt.show()
        
            # plt.imshow(ig)
            # plt.plot(xcord,ycord,'red')
            # plt.title("Before matched filtering")
            # plt.show()
            
            # plt.imshow(ig)
            # plt.plot(newx,newy,'red')
            # plt.title("After matched filtering")
            # plt.show()
            return np.array(newx), np.array(newy)
    else:
        if((len(newx) == 0 or mpeak < thresh) and flag == 0):
            return line(theta,ig.shape[0]-1-rho,ig,1,low_snr=low_snr,debug=debug)
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

# Changes made from dtype = float to int  
  
    bkg_image = bkg.back(dtype=int)
    image_subtracted = image - bkg_image

    thresh = 50 if low_snr else 200

    objects = []
    try:
        objects = sep.extract(image_subtracted, thresh)
    except:
        # print(f"Error at Source Extraction: Failed to get sources at thresh {thresh} attempting thresh {thresh+5}")
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

    while start < limit:
        img = data[jump*start:jump*(start+1),:]
        rho, theta = fast_radon(img, 7)

        tt = math.floor(np.abs((theta*5/10)-180))
        if tt in range(0,3) or tt in range(87,93) or tt in range(177,183):
            start += 1
            continue

        xx, yy = line((theta*5/10)-180, rho, img, low_snr=low_snr)
        # print(len(xx), low_snr)
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
            
            if len(prev_x) > len(xx):
                try:
                    points = np.column_stack((prev_x, prev_y))
                    distance_threshold = 2  # Set based on data scale
                    min_neighbors = 5
                    # dist_matrix = distance_matrix(points, points) 
                    # dense_mask = (dist_matrix < distance_threshold).sum(axis=1) > min_neighbors
                    # dense_points = points[dense_mask]
                    # x_dense, y_dense = dense_points[:, 0], dense_points[:, 1]

                    kdtree = KDTree(points)

                    # Find the number of neighbors within the distance threshold
                    neighbors_count = np.array([
                        len(kdtree.query_ball_point(point, distance_threshold)) for point in points
                    ])

                    # Apply the density filter
                    dense_mask = neighbors_count > min_neighbors
                    dense_points = points[dense_mask]

                    # Extract dense x and y points
                    x_dense, y_dense = dense_points[:, 0], dense_points[:, 1]
                except:
                    x_dense, y_dense = [], []

                if len(x_dense) > 0:
                    xx = np.array(x_dense)
                    yy = np.array(y_dense)
                else:
                    xx = np.array(prev_x)
                    yy = np.array(prev_y)
                f = 0
                coefficients = np.polyfit(xx, yy, 1)
                tyy = np.polyval(coefficients, xx)
                f = 0
                while st > jump*(start+1):
                    start += 1
                    f = 1
                
                if f == 1:
                    start -= 1
            
            xfinal.append(xx)
            if len(tyy) > 0:
                yfinal.append(tyy)
            else:
                yfinal.append(yy)
        start += 1
    return xfinal, yfinal


def generate_stats(x,y, headers):
    # points = np.column_stack((x, y))
    # centroid = np.mean(points, axis=0)
    # centered_points = points - centroid
    # _, _, vh = np.linalg.svd(centered_points)
    # direction_vector = vh[0]
    # projections = centered_points @ direction_vector
    # min_proj, max_proj = projections.min(), projections.max()
    # start_point = centroid + min_proj * direction_vector
    # end_point = centroid + max_proj * direction_vector
    # angle_radians = np.arctan2(direction_vector[1], direction_vector[0])
    # angle_degrees = np.degrees(angle_radians)
    # line_length = np.abs(np.linalg.norm(end_point - start_point))

    chunk_size = 10000

    centroid_sum = np.zeros(2, dtype=np.float32)
    count = 0
    for start_idx in range(0, len(x), chunk_size):
        end_idx = start_idx + chunk_size
        chunk_x = np.array(x[start_idx:end_idx], dtype=np.float32)
        chunk_y = np.array(y[start_idx:end_idx], dtype=np.float32)
        centroid_sum += np.array([chunk_x.sum(), chunk_y.sum()], dtype=np.float32)
        count += len(chunk_x)
    centroid = centroid_sum / count

    # Subtract the centroid to center the points (chunk-wise)
    centered_points = []
    for start_idx in range(0, len(x), chunk_size):
        end_idx = start_idx + chunk_size
        chunk_x = np.array(x[start_idx:end_idx], dtype=np.float32)
        chunk_y = np.array(y[start_idx:end_idx], dtype=np.float32)
        chunk_points = np.column_stack((chunk_x, chunk_y)) - centroid
        centered_points.append(chunk_points)
    centered_points = np.vstack(centered_points)  # Combine all chunks

    # Apply PCA to find the best-fit line direction
    pca = PCA(n_components=1)
    pca.fit(centered_points)
    direction_vector = pca.components_[0]

    # Project points onto the line to find the start and end points
    projections = centered_points @ direction_vector
    min_proj, max_proj = projections.min(), projections.max()
    start_point = centroid + min_proj * direction_vector
    end_point = centroid + max_proj * direction_vector

    # Calculate the angle of the line
    angle_radians = np.arctan2(direction_vector[1], direction_vector[0])
    angle_degrees = np.degrees(angle_radians)

    # Calculate the length of the line
    line_length = np.linalg.norm(end_point - start_point)

    # print("centroid: ", centroid)
    # print("start: ", start_point)
    # print("end: ", end_point)
    # print("orientation: ", angle_degrees)
    # print("length: ", line_length)

    plt.imshow(image, vmin=0, vmax=255, cmap='gray')
    plt.plot(int(centroid[0]), int(centroid[1]), label="Centroid", marker='x')
    plt.plot(int(start_point[0]), int(start_point[1]), label="Start", marker='x')
    plt.plot(int(end_point[0]), int(end_point[1]), label="End", marker='x')
    plt.show()


    start_time = dt.strptime(headers['DATE-OBS']+" "+headers['UTSTART '], '%Y-%m-%d %H:%M:%S.%f')
    wcs = WCS(headers)
    pixel_scale = wcs_lib.utils.proj_plane_pixel_scales(wcs)
    # size = (pixel_scale[0]*3600*image.shape[1], pixel_scale[1]*3600*image.shape[0])
    angular_speed = 15
    delta_t = (pixel_scale[1]*3600)/angular_speed
    centroid_ra, centroid_dec = wcs.wcs_pix2world(int(centroid[0]), int(centroid[1]), 1)
    t_at_cen = start_time + datetime.timedelta(seconds=int(centroid[1])*delta_t)
    t_at_end = start_time + datetime.timedelta(seconds=int(end_point[1])*delta_t)
    t_at_start = start_time + datetime.timedelta(seconds=int(start_point[1])*delta_t)
    if t_at_start > t_at_end:
        temp_t = t_at_start
        t_at_start = t_at_end
        t_at_end = temp_t
        temp_st = start_point
        start_point = end_point
        end_point = temp_st
    dt_streak = t_at_end-t_at_start
    end_time =  start_time + datetime.timedelta(seconds=int(headers['NAXIS2'])*delta_t)
    return {
        "centroid" : [centroid_ra, centroid_dec],
        "tac" : t_at_cen.strftime('%d/%m/%Y %H:%M:%S.%f'),
        "duration" : dt_streak.total_seconds(),
        "start" : [int(start_point[0]), int(start_point[1])],
        "end" : [int(end_point[0]), int(end_point[1])],
        "orientation" : np.round(angle_degrees,2),
        "length" : int(line_length),
        "pixel_scale_x" : pixel_scale[0]*3600,
        "pixel_scale_y" : pixel_scale[1]*3600,
        "start_time" : t_at_start.strftime('%d/%m/%Y %H:%M:%S.%f'),
        "end_time" : t_at_end.strftime('%d/%m/%Y %H:%M:%S.%f'),
        "ut_end" : end_time.strftime('%d/%m/%Y %H:%M:%S.%f')
    }

def generate_stats_debug(x,y, image, headers):
    points = np.column_stack((x, y))
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    direction_vector = vh[0]
    projections = centered_points @ direction_vector
    min_proj, max_proj = projections.min(), projections.max()
    start_point = centroid + min_proj * direction_vector
    end_point = centroid + max_proj * direction_vector
    angle_radians = np.arctan2(direction_vector[1], direction_vector[0])
    angle_degrees = np.degrees(angle_radians)
    line_length = np.linalg.norm(end_point - start_point)

    print("centroid: ", centroid)
    print("start: ", start_point)
    print("end: ", end_point)
    print("orientation: ", angle_degrees)
    print("length: ", line_length)

    plt.imshow(image, vmin=0, vmax=255, cmap='gray')
    plt.scatter(x,y)
    plt.plot(int(centroid[0]), int(centroid[1]), label="Centroid", marker='x')
    plt.plot(int(start_point[0]), int(start_point[1]), label="Start", marker='x')
    plt.plot(int(end_point[0]), int(end_point[1]), label="End", marker='x')
    plt.show()


    start_time = dt.strptime(headers['DATE-OBS']+" "+headers['UTSTART '], '%Y-%m-%d %H:%M:%S.%f')
    wcs = WCS(headers)
    pixel_scale = wcs_lib.utils.proj_plane_pixel_scales(wcs)
    # size = (pixel_scale[0]*3600*image.shape[1], pixel_scale[1]*3600*image.shape[0])
    angular_speed = 15
    delta_t = pixel_scale[1]/angular_speed
    centroid_ra, centroid_dec = wcs.wcs_pix2world(int(centroid[0]), int(centroid[1]), 1)
    t_at_cen = start_time + datetime.timedelta(0, int(centroid[1])*delta_t)
    t_at_end = start_time + datetime.timedelta(0,int(end_point[1])*delta_t)
    t_at_start = start_time + datetime.timedelta(0,int(start_point[1])*delta_t)
    dt_streak = t_at_end-t_at_start

    return {
        "centroid" : [centroid_ra, centroid_dec],
        "tac" : t_at_cen.strftime('%d/%m/%Y %H:%M:%S.%f'),
        "duration" : dt_streak.total_seconds(),
        "start" : [int(start_point[0]), int(start_point[1])],
        "end" : [int(end_point[0]), int(end_point[1])],
        "orientation" : np.round(angle_degrees,2),
        "length" : int(line_length)
    }

def generate_output(path):
    dates = [x[0] for x in os.walk(path)]

    for d in dates[1:]:
        output_df =  pd.DataFrame(columns=["Date", "Time", "Streak-Detected", "Orientation (Deg)", "Length (Pixels)", "Right Ascension", "Declination", "Time at Centroid", "Delta-T (Seconds)", "Start (x,y)", "End (x,y)", "UT Time at start (streak)", "UT Time at end (streak)", "Pixel scale x", "Pixel scale y", "UT Start Time (image)", "UT End Time (image)"])
        parent_directory = f"{d}\\*.fits"
        lsf = glob.glob(parent_directory)
        for sf in lsf:
            print("Current File: ",sf)
            st = time.time()
            headers, image = read_fits_file(sf)
            start_time = dt.strptime(headers['DATE-OBS']+" "+headers['UTSTART '], '%Y-%m-%d %H:%M:%S.%f')
            f = 0
            lx, ly = detect_line(image,low_snr=True)
            if len(lx) == 0:
                f = 1
            else:
                for x,y in zip(lx,ly):
                    stats = generate_stats(x,y,headers)
                    output_df.loc[len(output_df)] = [sf.split("\\")[-2], sf.split("\\")[-1],"YES",stats["orientation"],stats["length"],stats["centroid"][0],stats["centroid"][1],stats["tac"],stats["duration"],stats["start"],stats["end"],stats["start_time"],stats["end_time"],stats["pixel_scale_x"],stats["pixel_scale_y"],start_time, stats["ut_end"]]
            hx, hy = detect_line(image,low_snr=False)
            et = time.time()
            print(f'TIME TAKEN: {et-st} Seconds')
            return
            if len(hx) == 0 and f == 1:
                wcs = WCS(headers)
                pixel_scale = wcs_lib.utils.proj_plane_pixel_scales(wcs)
                angular_speed = 15
                delta_t = (pixel_scale[1]*3600)/angular_speed
                end_time =  start_time + datetime.timedelta(0,int(headers['NAXIS2'])*delta_t)
                output_df.loc[len(output_df)] = [sf.split("\\")[-2], sf.split("\\")[-1],"NO",0,0,0,0,"-",0,[-1,-1],[-1,-1],"-","-",-1,-1,start_time, end_time]
            else:
                for x,y in zip(hx,hy):
                    stats = generate_stats(x,y,headers)
                    output_df.loc[len(output_df)] = [sf.split("\\")[-2], sf.split("\\")[-1],"YES",stats["orientation"],stats["length"],stats["centroid"][0],stats["centroid"][1],stats["tac"],stats["duration"],stats["start"],stats["end"],stats["start_time"],stats["end_time"],stats["pixel_scale_x"],stats["pixel_scale_y"],start_time, stats["ut_end"]]
            gc.collect()
        return
        # o_path = f"C://Users//omar_//OneDrive//Desktop//streaks-final//output//actual//output_{sf.split("//")[-2]}.csv"
        # os.makedirs(os.path.dirname(o_path), exist_ok=True)
        # output_df.to_csv(o_path, index=False, mode='w')






# headers, image = read_fits_file("D:\\desktop\\wfh\\corrected_images\\24-10-2022\\20221024_r_1h08m.fits")
# # headers, image = read_fits_file("D:\\desktop\\wfh\\corrected_images\\01-11-2022\\20221101_r_6h37m.fits")
# xx, yy = detect_line(image, low_snr=False)
# for x,y in zip(xx,yy):
#     stats = generate_stats_debug(x,y,get_scaled_image(image),headers)

#     print(stats)

# path = "D:\\desktop\\wfh\\corrected_images"

# generate_output(path=path)


path = "E:\\wfh\\streak_obj_association\\DATA\\images\\20230513\\20230513_r_13h46m.fits"
headers, image = read_fits_file(path)
# start = 2
# jump = image.shape[1]
# scaled_image = get_scaled_image(image.copy())
# scaled_image = scaled_image[jump*start:jump*(start+1),:]
# # src_img = get_source_extracted_image(scaled_image, True)
# # rho, theta = fast_radon(src_img, 7)
# # xx, yy = line((theta*5/10)-180, rho, src_img, low_snr=True, debug=True)
# plt.imshow(scaled_image, cmap='gray',vmin=0, vmax=255, origin='lower')
# # plt.plot(xx,yy)
# plt.show()

lx, ly = detect_line(image,low_snr=True)
print(len(lx))
print(lx)
for x,y in zip(lx,ly):
    stats = generate_stats(x,y,headers)