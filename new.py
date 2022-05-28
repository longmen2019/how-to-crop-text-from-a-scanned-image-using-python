# cv2 is the module import name for opencv-python, "Unofficial pre-built CPU-only OpenCV packages for Python."
# The traditional OpenCV has many complicated steps involving building the module from scratch, which is unnecessary.
# Opencv is an open source library which is very useful for computer vision application such as video analysis,
#  CCTV footage analysis and image analysis.
# OpenCV is written by C++ and has more than 25,000 optimized algorithms.
# When we create applications for computer vision that we don't want to build from scratch we can use this library
# to start focusing on real world problems.
import cv2
import numpy as np

# Read image
# This is the Syntax to read the image
img = cv2.imread('receipt.jpg')

# It provides the shape value of an image, like height and width pixel values.
# Now you can print the image shape using print method
hh, ww = img.shape[:2]

# get edges
# https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
# OpenCv puts all the above in single function, cv.Canny().
# We will see how to use it
# First argument is our input image. Second and third arguments are our minVal and maxVal respectively
# Fourth argument is aperture_size. It is the size of Sobel kernel used for find image gradient.
# By default it is 3. Last argument is L2gradient which specifies the equation for finding gradient magnitude.
# If it is True, it uses the equation mentioned above which is more accurate, otherwise it uses this function:
# Edge_Gradient (G) = |Gx| + |Gy|. By default, it is False

canny = cv2.Canny(img, 50, 200)


# To put in simple words findContours detects change in the image color and marks it as contour.
# As example, the image of number written on paper the number would be detected as contour
# It is the method in which you want to store to the Contours.
# If set to CHAIN_APPROX_NONE stores absolutely all the contour points
# If set to CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points
# with CHAIN_APPROX_NONE length of contours stored is 524 i.e all points are stored in array
# With CHAIN_APPROX_SIMPLE length of contours stored is 4 i.e only corner four points are stored in array
contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# filter out small regions
# numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
cimg = np.zeros_like(canny)
# array_like: The shape and data-type of a define these same attributes of the returned array.
for cntr in contours:
    # Contours are defined as the line joining all the points along the boundary of an image that are having the same intensity
    # Contours come handy in shape analysis, finding the size of the object of interest, and object detection.
    # OpenCV has findContour() function that helps in extracting the contours from the image.
    # It works best on binary images, so we should first apply thresholding techniques, Sobel edges, etc.
    area = cv2.contourArea(cntr)
    if area > 20:
        # Using contour detection, we can detect the borders of objects, and localize them easily in an image.
        # It is often the first step for many interesting applications, such as image-foreground extraction
        cv2.drawContours(cimg, [cntr], 0, 255, 1)

# get convex hull and draw on input
# numpy.column_stack(tup): Stack 1-D arrays as columns into a 2-D array.
# Take a sequence of 1-D arrays and stack them as columns to make a single 2-D array. 2-D arrays are stacked as-is
# Just like with hstack. 1-D arrays are turned into 2-D columns first
# Parameters: tup: sequence of 1-D or 2-D arrays. Arrays to stack. All of them must have the same first dimension.
points = np.column_stack(np.where(cimg.transpose() > 0))
# A Convex object is one with no interior angles greater than 180 degrees.
# A shape that is not convex is called Non-Convex or Concave.
# Hull means the exterior or the shape of the object.
# Therefore, the Convex Hull of a shape or a group of points is a tight fitting convex boundary around the ponts or the shape
# The Convex Hull of a convex object is simply its boundary. The Convex Hull of a concave shape is a convex boundary that most tightly encloses it
# Gift Wrapping Algorithms : Given a set of points that define a shape, how do we find its convex hull?
# The algorithms for finding the Convext Hull are often called Gift Wrapping algorithms.
hull = cv2.convexHull(points)
himg = img.copy()
cv2.polylines(himg, [hull], True, (0, 0, 255), 1)

# draw convex hull as filled mask
# numpy.zeros_like(a, dtype=None, order='k', subok=True, shape=None)
# Return an array of zeros with the same shape and type as a given array
mask = np.zeros_like(cimg, dtype=np.uint8)
cv2.fillPoly(mask, [hull], 255)

# blacken out input using mask
mimg = img.copy()
# Whenever we are dealing with images while solving computer vision problems,
# there arises a necessity to wither manipulate the given image or extract parts of the given image based on the requirement,
# in such cases we make use of bitwise operators in OpenCV and when the elements of the arrays corresponding to the given two images must be combined bit wise,
# then we make use of an operator in OpenCV called but wise and operator using
# which the arrays corresponding to the two images can be combined resulting in merging of the two images
# and bit wise operation on the two images returns an image with the merging done as per the specification
# bitwise_and(source1_array, source2_array, destination_array, mask)
# where source1_array is the array corresponding to the first input image on which bitwise and operation is to be performed
# source2_array is the array corresponding to the second input image on which bitwise and operation is to be performed,
# destination_array is the resulting array by performing bitwise operation on the array corresponding to the first input image and the array corresponding to the second input image and
# mask is the mask operation to be performed on the resulting image and it is optional
mimg = cv2.bitwise_and(mimg, mimg, mask=mask)

# get rotate rectangle
# As is clear from the name, the bounding rectangle is drawn with a minimum area. Because of this, rotation is also consided.
# The below image shows 2 rectangles, the green one is the normal bounding rectangle while thered one is the minimum area rectangle
# OpenCV provides a function cv2.minAreaReact() for finding the minimum area rotated rectangle. This takes as input a 2D point
# set and returns a Box2D structure which contains the following details-(center(x,y), (width,height), angle of rotation)
rotrect = cv2.minAreaRect(hull)
(center), (width, height), angle = rotrect
# BoxPoints(RotatedRect) : Find the four vertices of a rotated rect. Useful to draw the rotated rectangle.
# The function find the four vertices of a rotated rectangle. This function is useful to draw the rectangle.

box = cv2.boxPoints(rotrect)
# int0 is an alias for intp; this, in turn is integer used for indexing (same as C size_t; normally either int32 or int 64)
boxpts = np.int0(box)

# draw rotated rectangle on copy of input
rimg = img.copy()
cv2.drawContours(rimg, [boxpts], 0, (0, 0, 255), 1)

# from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle tends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
    angle = -(90 + angle)

# otherwise, check width vs height
else:
    if width > height:
        angle = -(90 + angle)

    else:
        angle = -angle

# negate the angle to unrotate
neg_angle = -angle
print('unrotation angle:', neg_angle)
print('')

# Get rotation matrix
# center = (width // 2, height // 2)
# cv2.getRotationMatrix2D() function is used to make the transformation matrix M which will be used for rotating image
# center: Center of rotation
# angle(0): Angle of Rotation, Angle is positive for anti-clockwise and negative for clockwise.
# scale: scaling factor which scales the image
M = cv2.getRotationMatrix2D(center, neg_angle, scale=1.0)

# unrotate to rectify
# In Affine transformation, all parallel lines in the original image will still be parallel in the output image
# To find the transformation matrix, we need three points from input image and their corresponding locations in the output image
# Then cv2.getAffineTransform will create 2x3 matrix which is to be passed to cv2.warpAffine.
result = cv2.warpAffine(mimg, M, (ww, hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# save results
cv2.imwrite('receipt_mask.jpg', mask)
cv2.imwrite('receipt_edges.jpg', canny)
cv2.imwrite('receipt_filtered_edges.jpg', cimg)
cv2.imwrite('receipt_hull.jpg', himg)
cv2.imwrite('receipt_rotrect.jpg', rimg)
cv2.imwrite('receipt_masked_result.jpg', result)

cv2.imshow('canny', canny)
cv2.imshow('cimg', cimg)
cv2.imshow('himg', himg)
cv2.imshow('mask', mask)
cv2.imshow('rimg', rimg)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()