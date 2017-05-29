import numpy as np
import cv2

# Display tools
import IPython.display
import PIL.Image

# Write tools
from cStringIO import StringIO


## Constants

left_lane_verts = np.array([[[0, 539], [358, 539], [470, 323], [421, 323]]])
right_lane_verts = np.array([[[960 - vert[0], vert[1]] for vert in left_lane_verts[0]]])


## Functions

def showarray(a, fmt='png'):
    '''
    Displays an image without the ugliness of matplotlib
    '''
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def show_thresholded_image(image, threshold = [0, 0, 0], show=True):
    color_select = np.copy(image)

    thresholds = (image[:,:,0] < threshold[0]) \
        | (image[:,:,1] < threshold[1]) \
        | (image[:,:,2] < threshold[2])

    color_select[thresholds] = [0, 0, 0]
    
    if show:
        showarray(color_select, fmt='png')
        
    return thresholds


def lane_line_hue(image, val_threshold=190):
    '''
    Takes in an RGB image, converts to HSV, isolates lane lines, and determines hue for each,
    and returns that value
    val_threshold gives the threshold for brightness/value to identify lane lines
    '''
    llane_masked = region_of_interest(image, left_lane_verts)
    llane_hsv = cv2.cvtColor(llane_masked, cv2.COLOR_RGB2HSV)

    rlane_masked = region_of_interest(image, right_lane_verts)
    rlane_hsv = cv2.cvtColor(rlane_masked, cv2.COLOR_RGB2HSV)

    l_mean = llane_hsv[llane_hsv[:,:,2] >= val_threshold][:,1].mean()
    r_mean = rlane_hsv[rlane_hsv[:,:,2] >= val_threshold][:,1].mean()

    return [l_mean, r_mean]


# LINE PROCESSING

def lane_line_color(mean_hue, hue_threshold=90):
    if mean_hue >= hue_threshold:
        return 'yellow'
    else:
        return 'white'


def get_lines(image, rho, theta, vote_threshold, min_line_length, max_line_gap):
    lines_unformatted = cv2.HoughLinesP(image, rho, theta, vote_threshold, np.array([]),
            min_line_length, max_line_gap)
    lines = [((line[0][0], line[0][1]), (line[0][2], line[0][3])) for line in lines_unformatted]
    
    return lines


def get_line_intercepts(p0, p1, y_bottom, y_top):
    x0, y0 = map(float, p0)
    x1, y1 = map(float, p1)

    x_bottom = 1.0 * x0 + (x1 - x0) * ((y_bottom - y0) / (y1 - y0))
    x_top = 1.0 * x0 + (x1 - x0) * ((y_top - y0) / (y1 - y0))
    
    return (x_bottom, y_bottom), (x_top, y_top)


def extend_line(line, y_bottom, y_top):
    p0, p1 = line
    
    p_bottom, p_top = get_line_intercepts(p0, p1, y_bottom, y_top)
    
    return p_bottom, p_top


def average_lines(lines):
    x_bottom = np.mean([line[0][0] for line in lines])
    y_bottom = np.mean([line[0][1] for line in lines])
    x_top = np.mean([line[1][0] for line in lines])
    y_top = np.mean([line[1][1] for line in lines])
    
    return (x_bottom, y_bottom), (x_top, y_top)


def int_line(line):
    intify = lambda x: int(round(x))
    return tuple(map(intify, line[0])), tuple(map(intify, line[1]))


## Classes

class LanesImage(object):
    # TODO - make this a class that applies to either left or right, and then each image has two
    def __init__(self, original_image):
        self.original_image = np.copy(original_image)
        self.l_verts = left_lane_verts
        self.r_verts = right_lane_verts


    def mask_lanes(self):
        self.l_masked = region_of_interest(self.original_image, self.l_verts)
        self.r_masked = region_of_interest(self.original_image, self.r_verts)
        
        return self.l_masked, self.r_masked


    def get_hue_and_color(self, val_threshold=190):
        self.l_masked_hsv, self.r_masked_hsv = \
            map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV), [self.l_masked, self.r_masked])

        self.l_hue = self.l_masked_hsv[self.l_masked_hsv[:,:,2] >= val_threshold][:,1].mean()
        self.r_hue = self.r_masked_hsv[self.r_masked_hsv[:,:,2] >= val_threshold][:,1].mean()

        self.l_color, self.r_color = map(lane_line_color, [self.l_hue, self.r_hue])
        
        return ((self.l_hue, self.l_color), (self.r_hue, self.r_color))


    def bw_lanes(self, val_threshold=150, hue_threshold=70):
        self.l_bw = np.copy(self.l_masked)

        if self.l_color == 'white':
            val_thresholds = self.l_masked_hsv[:,:,2] > val_threshold
            self.l_bw[val_thresholds] = [255,255,255]
            self.l_bw[~val_thresholds] = [0,0,0]

        elif self.l_color == 'yellow':
            val_thresholds = self.l_masked_hsv[:,:,2] > val_threshold
            sat_thresholds = self.l_masked_hsv[:,:,1] > hue_threshold
            self.l_bw[val_thresholds & sat_thresholds] = [255, 255, 255]
            self.l_bw[~(val_thresholds & sat_thresholds)] = [0, 0, 0]


        self.r_bw = np.copy(self.r_masked)

        if self.r_color == 'white':
            val_thresholds = self.r_masked_hsv[:,:,2] > val_threshold
            self.r_bw[val_thresholds] = [255,255,255]
            self.r_bw[~val_thresholds] = [0,0,0]

        elif self.r_color == 'yellow':
            val_thresholds = self.r_masked_hsv[:,:,2] > val_threshold
            sat_thresholds = self.r_masked_hsv[:,:,1] > hue_threshold
            self.r_bw[val_thresholds & sat_thresholds] = [255, 255, 255]
            self.r_bw[~(val_thresholds & sat_thresholds)] = [0, 0, 0]


    def gaussian_blur(self, target=None, kernel_size=7):
        target = target or [self.l_bw, self.r_bw]
        blur = lambda x: cv2.GaussianBlur(x, (kernel_size, kernel_size), 0)
        self.l_gblur, self.r_gblur = map(blur ,target)


    def canny_edge_detection(self, target=None, canny_threshold=70):
        target = target or [self.l_gblur, self.r_gblur]
        canny = lambda x: cv2.Canny(x, canny_threshold, 2 * canny_threshold)
        self.l_canny, self.r_canny = map(canny, target)


    def line_candidates(self,
                        target=None,
                        rho=1,
                        theta = np.pi / 360,
                        vote_threshold = 35,
                        min_line_length = 120,
                        max_line_gap = 110
                       ):
        target = target or [self.l_canny, self.r_canny]
        get_lines_params = lambda x: get_lines(x, rho, theta, vote_threshold,
            min_line_length, max_line_gap)
        self.l_line_candidates, self.r_line_candidates = map(get_lines_params, target)


    def smooth_lines(self, target=None, lines_buffer=None, lines_buffer_len_threshold=7):
        '''Smooths out line candidates.  lines_buffer stores the previous values of smooth_lines for previous frames.

        If lines_buffer is present, smooth_lines will add the new smoothed line to the tail of lines_buffer, and remove the first element of lines_buffer if lines_buffer's length exceeds the lines_buffer_len_threshold.

        It then recomputes self.l_avg_line and self.r_avg_line based on the buffer values, throwing out the two outlier values to insulate line drawing against momentary errors.  It does this at the y_top and y_bottom intercepts irrespective of which line is which, and how they relate to one another, because I am lazy.
        '''
        l_lines, r_lines = target or [self.l_line_candidates, self.r_line_candidates]

        y_bottom = left_lane_verts[0][0][1]
        y_top = left_lane_verts[0][2][1]
        self.l_cand_extended = [extend_line(line, y_bottom, y_top) for line in l_lines]
        self.r_cand_extended = [extend_line(line, y_bottom, y_top) for line in r_lines]

        self.l_avg_line = int_line(average_lines(self.l_cand_extended))
        self.r_avg_line = int_line(average_lines(self.r_cand_extended))

        if lines_buffer:
            l_lines, r_lines = lines_buffer
            
            l_lines.append(self.l_avg_line)
            r_lines.append(self.r_avg_line)

            if len(l_lines) > lines_buffer_len_threshold:
                l_lines.pop(0)
                r_lines.pop(0)

            # get the intercepts for left and right lines
            l_bottom_values = [line[0][0] for line in l_lines]
            l_top_values = [line[1][0] for line in l_lines]
            r_bottom_values = [line[0][0] for line in r_lines]
            r_top_values = [line[1][0] for line in r_lines]

            # throw out the outliners
            if len(l_lines) == lines_buffer_len_threshold:
                l_bottom_values.sort()
                l_top_values.sort()
                r_bottom_values.sort()
                r_top_values.sort()

                l_bottom_values = l_bottom_values[1:-1]
                l_top_values = l_top_values[1:-1]
                r_bottom_values = r_bottom_values[1:-1]
                r_top_values = r_top_values[1:-1]

            l_lines = [((l_bottom_values[i], y_bottom), (l_top_values[i], y_top)) \
                for i in range(len(l_bottom_values))]
            r_lines = [((r_bottom_values[i], y_bottom), (r_top_values[i], y_top)) \
                for i in range(len(r_bottom_values))]

            self.l_avg_line = int_line(average_lines(l_lines))
            self.r_avg_line = int_line(average_lines(r_lines))


    def draw_lines(self, base=None, lines=None, color=[[234, 2, 80], [234, 2, 80]],
                   transparency = 0.5, width=6):
        base = base or self.original_image
        self.line_image = np.zeros_like(self.original_image)
        lines = lines or [self.l_avg_line, self.r_avg_line]
        
        for i, line in enumerate(lines):
            cv2.line(self.line_image, tuple(line[0]), tuple(line[1]), color[i], width)

        replace = self.line_image != [0,0,0]
        base[replace] = self.line_image[replace]

        self.base_w_lines = base


    def create_lane_lines(self, lines_buffer=None, lines_buffer_len_threshold=7, debug=False):
        self.mask_lanes()
        self.get_hue_and_color()
        self.bw_lanes()
        self.gaussian_blur()
        self.canny_edge_detection()
        if debug:
            showarray(self.l_canny)
            showarray(self.r_canny)

        self.line_candidates()
        if debug:
            print self.l_line_candidates
            print self.r_line_candidates

        self.smooth_lines(target=None,
                          lines_buffer=lines_buffer,
                          lines_buffer_len_threshold=lines_buffer_len_threshold)

        self.draw_lines()

        return self.base_w_lines