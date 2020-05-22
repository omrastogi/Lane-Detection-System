from matplotlib import pyplot as plt
import numpy as np
import cv2

lpnts = []
rpnts = []
def show(cells):
	cv2.imshow('img',cells)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_perspective(img):
	dst_size=(1280,720)
	h,w= np.shape(img)
	dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])
	src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])
	img_size = np.float32([(img.shape[1],img.shape[0])])
	pts2= dst * np.float32(dst_size)
	pts1=src*img_size
	M = cv2.getPerspectiveTransform(pts1, pts2)
	warped = cv2.warpPerspective(img, M, dst_size)
	return warped

def inv_perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def draw_lane(img,lane):
	histogram = np.sum(img,axis = 0)
	midpoint = int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint


	nwindows = 9
	window_height = np.int(img.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the imag
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for 

	leftx_current = leftx_base
	rightx_current = rightx_base

	draw_windows = True
	margin = 150
	minpix = 1
	left_lane_inds = []
	right_lane_inds = []

	left_a, left_b, left_c = [],[],[]
	right_a, right_b, right_c = [],[],[]


	for window in range(nwindows):
		win_y_low = img.shape[0] - (window+1)*window_height
		win_y_high = img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		if draw_windows == True:
			cv2.rectangle(img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
			(100,255,255), 3) 
			cv2.rectangle(img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
			(100,255,255), 3) 
		
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
		
	# show(img)


	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	color_img = np.zeros_like(lane)



	left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

	if len(rpnts)<2:
		rpnts.append(right)
		lpnts.append(left)
	if len(rpnts)==2:
		right = (right+rpnts[0]+rpnts[1])/3
		left = (left+lpnts[0]+lpnts[1])/3
		rpnts[0] = rpnts[1]
		rpnts[1] = right
		lpnts[0] = lpnts[1]
		lpnts[1] = left

	points = np.hstack((left, right))


	for pt in left[0]:
		cv2.circle(color_img ,(int(pt[0]),int(pt[1])), 7, (10,200,10), -1)
	for pt in right[0]:
		cv2.circle(color_img ,(int(pt[0]),int(pt[1])), 7, (10,200,10), -1)

	cv2.fillPoly(color_img, np.int_(points), (255,10,10))
	h,w,c = np.shape(lane)
	inv = inv_perspective_warp(color_img,dst_size=(w,h))
	img = cv2.addWeighted(inv, 0.5,lane,1,0)

	return (img)
#----------------------------------------------

def load(lane):	
	orig = lane.copy()

	lane = cv2.resize(lane,(1280,720),cv2.INTER_NEAREST)
	h,w,c = np.shape(lane)

	blur = cv2.blur(lane, (10,10))



	hls = cv2.cvtColor(lane, cv2.COLOR_RGB2HLS).astype(np.float)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	binary = l_channel

	sobelx = cv2.Sobel(binary, cv2.CV_64F, 1,1, ksize=1)
	abs_sobelx = np.absolute(sobelx)
	
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))


	perspect = get_perspective(scaled_sobel)


	binary = cv2.threshold(perspect, 15, 255, cv2.THRESH_BINARY)
	binary = np.array(binary[1])
	# show (binary)
	img = binary
	
	return (draw_lane(img, lane))

#------------------------------------------------------------------------------------#


cap = cv2.VideoCapture("challenge.mp4")
cnt = 0
while(1):
	ret, frame = cap.read()
	if ret == 0:
		break 
	
	if frame.max()<20:
		continue	
	
	img = load(frame)
	cv2.imshow("frame", img)
	if cv2.waitKey(1) == 27:  ## 27 - ASCII for escape key
		break
######
cap.release()
cv2.destroyAllWindows()