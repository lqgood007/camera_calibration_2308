import os
import cv2
import numpy as np
import cv2 as cv

def affine_detect(detector, img, mask=None):
    params = [(1.0, 0.0)]
    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))


    keypoints, descrs = [], []
    ires = [f(p) for p in params]
    for i, (k, d) in enumerate(ires):
        # print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)
    return keypoints, np.array(descrs)

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def anorm2(a):
    return (a*a).sum(-1)

def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai


def get_flann_matrix(imgs_bev_path = r"C:\work\TCSDetection\tcs\tcs_bev",img = r"C:\work\TCSDetection\tcs\images\2394333593.jpg"):#/media/lee/2T1/dataset/tcs/images
    imgs_bev = []
    img2 = cv2.imread(img,0)
    for r,d,files in os.walk(imgs_bev_path):
        for f in files:
            imgs_bev.append(os.path.join(imgs_bev_path,f))
    img_M = None
    detector = cv2.xfeatures2d.SIFT_create()
    flann_params = dict(algorithm=1, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    for img_bev in imgs_bev:
        img1 = cv2.imread(img_bev,0)
        kp1, desc1 = affine_detect(detector, img1)
        kp2, desc2 = affine_detect(detector, img2)
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= 10:
            H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
            try:
                M_T = np.linalg.inv(H) # juzhen qiu ni
                img_M = M_T
            except:
                print(H)
                return None
            # print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        else:
            img_M = None
            H, status = None, None
            print('%d matches found, not enough for homography estimation  @ %s' % (len(p1),img))
        # _vis = explore_match(str(len(p1))+"_"+str(len(p2))+"_"+img.split('/')[-1][:-4]+"_"+img_bev.split('/')[-1][:-4], img1, img2, kp_pairs, status, H)
    return img_M

def get_flann_matrix_img2img(img_bev = r"C:\work\TCSDetection\tcs\tcs_bev\test1.jpg",img = r"C:\work\TCSDetection\tcs\images\2394333593.jpg"):#/media/lee/2T1/dataset/tcs/images
    imgs_bev = []
    img_bev = cv2.imread(img_bev,0)
    img2 = cv2.imread(img,0)
    detector = cv2.xfeatures2d.SIFT_create()
    flann_params = dict(algorithm=1, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    kp1, desc1 = affine_detect(detector, img_bev)
    kp2, desc2 = affine_detect(detector, img2)
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    if len(p1) >= 10:
        H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
        try:
            M_T = np.linalg.inv(H) # juzhen qiu ni
            img_M = M_T
        except:
            print(H)
            return None
        # print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        img_M = None
        H, status = None, None
        print('%d matches found, not enough for homography estimation  @ %s' % (len(p1),img))
    return img_M


if __name__ == '__main__':
    m = get_flann_matrix_img2img()
    print(m)
    img = cv2.imread(r"C:\work\TCSDetection\tcs\images\2394333593.jpg",0)
    print(img.shape[0],img.shape[1])
    result = cv2.warpPerspective(img,m,(img.shape[0],img.shape[1]))
    cv2.imshow("i",result)
    cv2.waitKey(0)
    print(m)