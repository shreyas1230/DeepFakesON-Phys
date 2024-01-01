
import numpy as np
import os
import cv2
from tqdm import tqdm
import imutils
from imutils.video import FPS
np.seterr(all="raise")
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import cv2
import multiprocessing
import os
import sys
import threading
import tensorflow as tf
from IPython import embed


def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
    sys.stdout.flush()  # flush to stdout


def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(frame))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=False, every=1, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame, also handles case chunk_size < total

    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            print_progress(i, len(frame_chunks), prefix=prefix_str, suffix='Complete')  # print it's progress

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames

image_path = '../../data/Celeb-DF-v2'#'D:\\Pattern_Letters_HR_PAD\\BBDD\\3DMAD\\session03\\'
image_name_video = []
# Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
epsilon = 1e-6

# print(os.listdir(image_path))
# for root, folders, files in os.walk(image_path):
#     print(i,j,k)
#     for f in files:
#         if '.mp4' not in f:
#             continue
#         ruta_parcial = os.path.join(root, 'DeepFrames', f)
#         ruta_parcial2 = os.path.join(root, 'RawFrames', f)
# raise Exception

# for f in [f for f in os.listdir(image_path)]:
#     if not("_C.avi" in f): #OULU
#         continue
resume = False
for root, folders, files in os.walk(image_path):
    print(root)
    for f in tqdm(files):
        if ".mp4" not in f:
            continue
        # if f in ["id0_0000.mp4","id0_0001.mp4","id0_0002.mp4"]:
        #     continue
        # if f == "id27_0005.mp4":
        #     resume = True
        # if not resume:
        #     continue
        
        print(f)
        vid_base_path = f.split('.mp4')[0]
        carpeta= os.path.join(root, f)
        # fps = FPS().start()
        # path=video_to_frames(video_path=carpeta, frames_dir='test_frames', overwrite=False, every=5, chunk_size=1000)
        # fps.stop()
        # print("[INFO] elasped time: {}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {}".format(fps.fps()))

        # fps = FPS().start()
        # cap = cv2.VideoCapture(carpeta)
        # i=0
        # while int(cap.get(7)) > i:
        #     _, frame = cap.read()
        #     cv2.imwrite(os.path.join('test_frames', 'cv', f"{i}.jpg"), frame)
        #     i+=1
        # fps.stop()
        # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # raise Exception
        cap = cv2.VideoCapture(carpeta)

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        nFrames = cap.get(7)
        max_frames = int(nFrames)
        ruta_parcial = os.path.join(root, 'DeepFrames') #os.path.join('D:\\Pattern_Letters_HR_PAD\\BBDD\\3DMAD\\DeepFrames',f) 
        if not(os.path.exists(ruta_parcial)) :
            os.makedirs(ruta_parcial)
            # ruta_parcial = os.path.join(ruta_parcial, f)
        ruta_parcial2 = os.path.join(root, 'RawFrames') #os.path.join('D:\\Pattern_Letters_HR_PAD\\BBDD\\3DMAD\\RawFrames',f) 
        if not(os.path.exists(ruta_parcial2)) :
            os.makedirs(ruta_parcial2)
            # ruta_parcial2 = os.path.join(ruta_parcial2, f)
        
        L = 36
        C_R=np.zeros((L,L,max_frames))
        C_G=np.zeros((L,L,max_frames))
        C_B=np.zeros((L,L,max_frames))
        
        D_R=np.zeros((L,L,max_frames))
        D_G=np.zeros((L,L,max_frames))
        D_B=np.zeros((L,L,max_frames))
        
        D_R2=np.empty((L,L,max_frames))
        D_G2=np.empty((L,L,max_frames))
        D_B2=np.empty((L,L,max_frames))
        
        medias_R = np.empty((L,L))
        medias_G = np.empty((L,L))
        medias_B = np.empty((L,L))
        
        desviaciones_R = np.empty((L,L))
        desviaciones_G = np.empty((L,L))
        desviaciones_B = np.empty((L,L))
        
        imagen = np.empty((L,L,3))
        
        medias_CR = np.empty((L,L))
        medias_CG = np.empty((L,L))
        medias_CB = np.empty((L,L))
        
        desviaciones_CR = np.empty((L,L))
        desviaciones_CG = np.empty((L,L))
        desviaciones_CB = np.empty((L,L))
        ka            = 1
        
        
        # fps = FPS().start() #TESTING

        lst = []
        for i in range(1, max_frames):
            _, frame = cap.read()
            lst.append([i-1, frame])

        # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        def helper(elem):
            idx, frame = elem
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # np.random.seed(0)
            # tf.random.set_seed(0)
            # cv2.setRNGSeed(0)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Sort detections by first entry to maintain reproducibility
            # print("HI", faces)
            # print(idx)
            # print(faces[:,0])
            if len(faces) == 0:
                print(f, idx, "No faces detected!")
                return (None, None)

            faces = faces[faces[:,0].argsort(), :]
            for (x, y, w, h) in faces:
                # face = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (L,L), interpolation = cv2.INTER_AREA)
            return (faces, face)
        # embed()
        # raise Exception
        with ThreadPoolExecutor(max_workers=32) as executor:
            results = executor.map(helper, lst)
        res = [(faces, r) for faces, r in results]
        # from IPython import embed; embed()
        if len(res) == 0:
            continue

        # embed()
        vid_faces = list(list(zip(*res))[1])

        # embed()

        # Original code just copies over the prev frame's face if no faces detected
        for i in range(len(vid_faces)):
            if vid_faces[i] is None:
                if i == 0:
                    vid_faces[i] = np.zeros((L,L,3))
                else:
                    vid_faces[i] = vid_faces[i-1]
        # embed()
        vid_faces = np.array(vid_faces) # t x h x w x c
        # embed()
        # face_dets = list(zip(*res))[0]
        a = np.zeros((L,L,max_frames))
        b = np.zeros((L,L,max_frames))
        c = np.zeros((L,L,max_frames))
        d = np.zeros((L,L,max_frames))
        e = np.zeros((L,L,max_frames))
        f = np.zeros((L,L,max_frames))
        a[:,:,1:] = np.transpose(vid_faces[:,:,:,0], (1,2,0))
        b[:,:,1:] = np.transpose(vid_faces[:,:,:,1], (1,2,0))
        c[:,:,1:] = np.transpose(vid_faces[:,:,:,2], (1,2,0))

        d[:,:,1:-1] = (a[:,:,2:] - a[:,:,1:-1]) / (a[:,:,2:] + a[:,:,1:-1] + epsilon)
        e[:,:,1:-1] = (b[:,:,2:] - b[:,:,1:-1]) / (b[:,:,2:] + b[:,:,1:-1] + epsilon)
        f[:,:,1:-1] = (c[:,:,2:] - c[:,:,1:-1]) / (c[:,:,2:] + c[:,:,1:-1] + epsilon)

        C_R = a
        C_G = b
        C_B = c
        D_R = d
        D_G = e
        D_B = f

        # fps.stop()
        # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # from IPython import embed; embed()
        # raise Exception

        # cap = cv2.VideoCapture(carpeta)
        # fps = FPS().start()
        # af=[]
        # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # while(cap.isOpened() and ka< max_frames):
        #     # print(max_frames, ka)
        #     # print(1)
        #     ret, frame = cap.read()
        #     # print(2)
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     # Detect faces
        #     np.random.seed(0)
        #     tf.random.set_seed(0)
        #     cv2.setRNGSeed(0)
        #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #     # Sort detections by first entry to maintain reproducibility
        #     faces = faces[faces[:,0].argsort(), :]
        #     # print(3)
        #     #rectangle around the faces
        #     af.append(faces)
        #     for (x, y, w, h) in faces:
        #         # face = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #         face = frame[y:y + h, x:x + w]
                
        
        #     face = cv2.resize(face, (L,L), interpolation = cv2.INTER_AREA)
        #     # cv2.imshow('img', face)
        #     # cv2.waitKey()
        #     C_R[:,:,ka] = face[:,:,0]
        #     C_G[:,:,ka] = face[:,:,1]
        #     C_B[:,:,ka] = face[:,:,2]
        #     # print(4)
            
        #     if ka > 1:
        #         # try:
        #         #     tmp=( C_R[:,:,ka] - C_R[:,:,ka-1] ) / ( C_R[:,:,ka] + C_R[:,:,ka-1])
        #         # except Exception as e:
        #         #     print(e)
        #         #     print(C_R[:,:,ka] - C_R[:,:,ka-1])
        #         #     print(C_R[:,:,ka] + C_R[:,:,ka-1])
        #         #     raise Exception
        #         # print(C_R[:,:,ka] + C_R[:,:,ka-1])
        #         D_R[:,:,ka-1] = ( C_R[:,:,ka] - C_R[:,:,ka-1] ) / ( C_R[:,:,ka] + C_R[:,:,ka-1] + epsilon);
        #         D_G[:,:,ka-1] = ( C_G[:,:,ka] - C_G[:,:,ka-1] ) / ( C_G[:,:,ka] + C_G[:,:,ka-1] + epsilon);
        #         D_B[:,:,ka-1] = ( C_B[:,:,ka] - C_B[:,:,ka-1] ) / ( C_B[:,:,ka] + C_B[:,:,ka-1] + epsilon);
        #     ka = ka+1
        #     fps.update()
        # fps.stop()
        # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # from IPython import embed; embed()
        # raise Exception
        
        
        print(5)
        medias_R = np.mean(D_R, axis=2)
        medias_G = np.mean(D_G, axis=2)
        medias_B = np.mean(D_B, axis=2)
        desviaciones_R = np.std(D_R, axis=2)
        desviaciones_G = np.std(D_G, axis=2)
        desviaciones_B = np.std(D_B, axis=2)
        # for i in range(0,L):
        #     for j in range(0,L):
        #         medias_R[i,j]=np.mean(D_R[i,j,:]) 
        #         medias_G[i,j]=np.mean(D_G[i,j,:]) 
        #         medias_B[i,j]=np.mean(D_B[i,j,:]) 
        #         desviaciones_R[i,j]=np.std(D_R[i,j,:]) 
        #         desviaciones_G[i,j]=np.std(D_G[i,j,:]) 
        #         desviaciones_B[i,j]=np.std(D_B[i,j,:]) 
        # from IPython import embed; embed()
        # raise Exception
        
        medias_CR = np.mean(C_R, axis=2)
        medias_CG = np.mean(C_G, axis=2)
        medias_CB = np.mean(C_B, axis=2)
        desviaciones_CR = np.std(C_R, axis=2)
        desviaciones_CG = np.std(C_G, axis=2)
        desviaciones_CB = np.std(C_B, axis=2)
        # for i in range(0,L):
        #     for j in range(0,L):
        #         medias_CR[i,j]=np.mean(C_R[i,j,:]) 
        #         medias_CG[i,j]=np.mean(C_G[i,j,:]) 
        #         medias_CB[i,j]=np.mean(C_B[i,j,:]) 
        #         desviaciones_CR[i,j]=np.std(C_R[i,j,:]) 
        #         desviaciones_CG[i,j]=np.std(C_G[i,j,:]) 
        #         desviaciones_CB[i,j]=np.std(C_B[i,j,:])      
        # from IPython import embed; embed()
        # raise Exception   
        
        D_R2 = (C_R - medias_CR[:,:,np.newaxis]) / (desviaciones_CR[:,:,np.newaxis] + 0.1)
        D_G2 = (C_G - medias_CG[:,:,np.newaxis]) / (desviaciones_CG[:,:,np.newaxis] + 0.1)
        D_B2 = (C_B - medias_CB[:,:,np.newaxis]) / (desviaciones_CB[:,:,np.newaxis] + 0.1)
        # for k in range(0,max_frames):
        #     D_R2[:,:,k] = (C_R[:,:,k] - medias_CR)/(desviaciones_CR+000.1)
        #     D_G2[:,:,k] = (C_G[:,:,k] - medias_CG)/(desviaciones_CG+000.1)
        #     D_B2[:,:,k] = (C_B[:,:,k] - medias_CB)/(desviaciones_CB+000.1)
        # from IPython import embed; embed()
        # raise Exception   
        

        # from IPython import embed; embed()
        imagen = np.array((D_R2, D_G2, D_B2))
        imagen = np.transpose(imagen, (1, 2, 0, 3))
        imagen = np.uint8(imagen)
        np.save(os.path.join(ruta_parcial2, f"{vid_base_path}.npy"), imagen, allow_pickle=True)
        # for k in range(0,max_frames):
            
        #     # imagen[:,:,0] = D_R2[:,:,k]
        #     # imagen[:,:,1] = D_G2[:,:,k]
        #     # imagen[:,:,2] = D_B2[:,:,k]

        #     # imagen= np.uint8(imagen)
        #     # embed()
        #     # raise Exception
        #     nombre_salvar= os.path.join(ruta_parcial2, str(k)+'.png') # Appearance Model
        #     # cv2.imwrite(nombre_salvar, imagen)
        #     cv2.imwrite(nombre_salvar, imagen[:,:,:,k])
            

        D_R = (D_R - medias_R[:,:,np.newaxis]) / (desviaciones_R[:,:,np.newaxis] + 0.1)
        D_G = (D_G - medias_G[:,:,np.newaxis]) / (desviaciones_G[:,:,np.newaxis] + 0.1)
        D_B = (D_B - medias_B[:,:,np.newaxis]) / (desviaciones_B[:,:,np.newaxis] + 0.1)
        # for k in range(0,max_frames):
            
        #     D_R[:,:,k] = (D_R[:,:,k] - medias_R)/(desviaciones_R+000.1)
        #     D_G[:,:,k] = (D_G[:,:,k] - medias_G)/(desviaciones_G+000.1)
        #     D_B[:,:,k] = (D_B[:,:,k] - medias_B)/(desviaciones_B+000.1)
        # from IPython import embed; embed()
        # raise Exception   
        
        imagen = np.array((D_R, D_G, D_B))
        imagen = np.transpose(imagen, (1, 2, 0, 3))
        imagen = np.uint8(imagen)
        np.save(os.path.join(ruta_parcial, f"{vid_base_path}.npy"), imagen, allow_pickle=True)
        # imagen = np.zeros((L,L,3))
        # for k in range(0,max_frames):
            
        #     # imagen[:,:,0] = D_R[:,:,k]
        #     # imagen[:,:,1] = D_G[:,:,k]
        #     # imagen[:,:,2] = D_B[:,:,k]
            
        #     # imagen= np.uint8(imagen)
        #     # from IPython import embed; embed()
        #     # raise Exception 

        #     nombre_salvar= os.path.join(ruta_parcial,str(k)+'.png') # Motion Model
        #     # cv2.imwrite(nombre_salvar, imagen)
        #     cv2.imwrite(nombre_salvar, imagen[:,:,:,k])          
            
        print(6)
        cap.release()
        cv2.destroyAllWindows()
print("Exiting...")
