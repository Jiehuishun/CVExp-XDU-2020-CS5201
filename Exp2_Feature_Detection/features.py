import math
from PIL import ImageGrab
import cv2
import numpy as np
import scipy
from PIL import Image

from scipy import ndimage, spatial
from scipy.ndimage import filters


import transformations

def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        '''
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        Ix = ndimage.sobel(srcImage,axis = 1,mode = 'reflect')
        Iy = ndimage.sobel(srcImage,axis = 0,mode = 'reflect')
        Ixy = ndimage.gaussian_filter(Ix*Iy,0.5,mode = 'reflect')
        Ixx = ndimage.gaussian_filter(Ix*Ix,0.5,mode = 'reflect')
        Iyy = ndimage.gaussian_filter(Iy*Iy,0.5,mode = 'reflect')
        harris_Matrix = np.zeros([height,width,2,2])
        harris_Matrix[:,:,0,0] = Ixx
        harris_Matrix[:,:,0,1] = Ixy
        harris_Matrix[:,:,1,0] = Ixy
        harris_Matrix[:,:,1,1] = Iyy

        for i in range(height):
            for j in range(width):
                harrisImage[i,j] = np.linalg.det(harris_Matrix[i,j,:,:])-0.1*np.trace(harris_Matrix[i,j,:,:])**2
                orientationImage[i,j] = np.degrees(np.arctan2(Iy[i,j],Ix[i,j]))
        #raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage,nums = 1e10):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        """
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        #regardless of the ANMS algorhim, the code below is ok for TODO 2.
        destImage = harrisImage >= ndimage.filters.maximum_filter(harrisImage,7)
        
        #raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return destImage
        """
        #I add a default argument to this function, it for the ANMS.
        destImage = np.zeros_like(harrisImage,np.bool)
        destNums = 0
        height,width = harrisImage.shape[0:2]
        for r in range(max(height,width)//2+1,6,-1):
            maximaMx = ndimage.filters.maximum_filter(harrisImage,r)
            #if you want to make sure a neighbour must have significantly higher strength for suppression to take place, you can time the harrisImage by 0.9.
            #localmaxMx = 0.9*harrisImage >= maximaMx
            #it will not pass the test yet.
            localmaxMx = harrisImage >= maximaMx
            localmaxNums = localmaxMx[localmaxMx != False].size
            if localmaxNums <= nums:
                destNums = localmaxNums
                destImage = localmaxMx
            elif localmaxNums > nums:
                for i in range(height):
                    for j in range(width):
                        if(destImage[i,j] == False and localmaxMx[i,j] == True):
                            destImage[i,j] = True
                            destNums += 1
                            if(destNums == nums):
                                return destImage 
        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f.size = 10
                f.pt = x,y
                f.angle = orientationImage[y,x]
                f.response = harrisImage[y,x]
                
                #raise Exception("TODO in features.py not implemented")
                # TODO-BLOCK-END

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        return detector.detect(image,None)

## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))
        
        #pad to the edge of each axis 
        grayImage = np.pad(grayImage,((2,2),(2,2)))
        
        for i, f in enumerate(keypoints):
            x, y = f.pt
            x, y = int(x), int(y)

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN
            desc[i] = grayImage[y:y+5,x:x+5].flatten()
            
            #raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN
            x,y = f.pt
            x,y = int(x),int(y)
            
            T1 = transformations.get_trans_mx(np.array([-x,-y,0]))
            R = transformations.get_rot_mx(0,0,-f.angle*np.pi/180)
            S = transformations.get_scale_mx(0.2,0.2,1) 
            T2 = transformations.get_trans_mx(np.array([4,4,0]))
            transMx3D =  T2@R@S@T1
            
            transMx[:,0:2] = transMx3D[:2,:2]
            transMx[:,2] = transMx3D[:2,3]
            
            #raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            if np.std(destImage)<=1e-5 :
                desc[i] = np.zeros_like(destImage).flatten()
            else:
                destImage = (destImage-np.mean(destImage))/np.std(destImage)
                desc[i] = destImage.flatten()
            
            
            
            #raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


"""class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        for i in range(len(desc1)):
            dist = np.array([desc1[i]])
            temp = scipy.spatial.distance.cdist(dist, desc2)
            sort_distance = sorted(temp[0])
            min = list(temp[0]).index(sort_distance[0])

            DM = cv2.DMatch()
            DM.queryIdx = i
            DM.trainIdx = min
            DM.distance = temp[0][min]
            matches.append(DM)

        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches
"""

class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        distanceMx = scipy.spatial.distance.cdist(desc1,desc2)
        distMin = list(np.argmin(distanceMx,axis = 1))
        for i in range(desc1.shape[0]):
            DM = cv2.DMatch()
            DM.queryIdx = i
            DM.trainIdx = int(distMin[i])
            DM.distance = distanceMx[i,distMin[i]]
            matches.append(DM)
        
        
        #raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN
        distanceMx = scipy.spatial.distance.cdist(desc1,desc2)
        distMin = list(np.argmin(distanceMx,axis = 1))
        for i in range(desc1.shape[0]):
                DM = cv2.DMatch()
                DM.queryIdx = i
                #Without type conversion. it may cause "SystemError: error return without exception set"
                DM.trainIdx = int(distMin[i])
                # mask to get the second minimum and count out the ratio score
                DM.distance = distanceMx[i,distMin[i]]/np.min(distanceMx[i][ distanceMx[i] != distanceMx[i,distMin[i]] ])
                matches.append(DM)
        
        
        #raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))




if __name__ == '__main__':
    # here to generate the harris.png for the yosemite1.jpg
    #image1 = np.array(Image.open('D:\\Semester5\\Computer Vision\\Exp2\\Exp2_Feature_Detection\\resources\\yosemite\\yosemite1.jpg'))
    #grayImage1 = cv2.cvtColor(image1.astype(np.float32)/255.0,cv2.COLOR_BGR2GRAY)
    #hkd = HarrisKeypointDetector()
    #kp = hkd.detectKeypoints(image1)


    
    image1 = np.array(Image.open('D:\\Semester5\\Computer Vision\\Exp2\\Exp2_Feature_Detection\\resources\\triangle1.jpg'))
    image2 = np.array(Image.open('D:\\Semester5\\Computer Vision\Exp2\\Exp2_Feature_Detection\\resources\\triangle1.jpg'))
    grayImage1 = cv2.cvtColor(image1.astype(np.float32)/255.0,cv2.COLOR_BGR2GRAY)
    grayImage2 = cv2.cvtColor(image2.astype(np.float32)/255.0,cv2.COLOR_BGR2GRAY)
    print(grayImage1.shape,grayImage2.shape)
    hkd1 = HarrisKeypointDetector()
    hkd2 = HarrisKeypointDetector()
    kp1 = hkd1.detectKeypoints(image1)
    kp2 = hkd2.detectKeypoints(image2)
    print(len(kp1),len(kp2))
    mopsfd1 = MOPSFeatureDescriptor()
    mopsfd2 = MOPSFeatureDescriptor()
    desc1 = mopsfd1.describeFeatures(image1,kp1)
    desc2 = mopsfd2.describeFeatures(image2,kp2)
    print(desc1.shape,desc2.shape)

    #print(desc1[desc1 != 0].shape,"\n--------------------------\n",desc2[desc2 != 0].shape)
    ssdfem1 = SSDFeatureMatcher()
    #ssdfem2 = mySSDFeatureMatcher()
    
    '''
    x = np.array([[2,1,7],[3,2,1],[2,5,2],[2,1,1],[4,6,2],[1,6,2]])
    y = np.array([[1,2,3],[3,3,3],[5,5,5]])

    mat1 = ssdfem1.matchFeatures(desc1,desc2)
    print("Done!")
    mat2 = ssdfem2.matchFeatures(x,y)
    for i in range(len(mat1)):
        print(mat1[i].queryIdx,mat1[i].trainIdx,mat1[i].distance)
    for i in range(len(mat2)):
        print(mat2[i].queryIdx,mat2[i].trainIdx,mat2[i].distance)
    '''
    



    #hkd = HarrisKeypointDetector()
    #image=np.array(Image.open('F:/course/cv/Exp2/Exp2/Exp2_Feature_Detection/resources/yosemite/yosemite1.jpg'))
    #grayImage=cv2.cvtColor(image.astype(np.float32)/255.0,cv2.COLOR_BGR2GRAY)
    # hars,ori= hkd.computeHarrisValues(grayImage)
    # print(np.max(ori))
    #  hlm = hkd.computeLocalMaxima(hars)
    #a = hkd.detectKeypoints(image)
    # print(a.__sizeof__())

