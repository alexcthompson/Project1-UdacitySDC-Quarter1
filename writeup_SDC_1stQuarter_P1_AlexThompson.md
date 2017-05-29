#**Finding Lane Lines on the Road** 

_Note: since I used Python 2.7 in my day job, I converted some functions to be Python 2.7 compatible.  The code in this notebook and the `tools.py` library should be compatible with Python 3, but if not, it should be straightforward to run it on a Python 2.7 kernel._

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

From the framework that Udacity provided I built a class, `LanesImage` that would organize the step by step process of identifying and drawing lane lines in the image.

My pipeline consists of the below steps, which are essentially listed out with minimal logic by the method `LanesImage.create_lane_lines`:

1. Load the image by initializing an instance of `LanesImage`,
2. Mask the lane line regions of interest with `LanesImage.mask_lanes`,
3. Determine the color of the lines with `LanesImage.get_hue_and_color`
4. Make BW masks of the lines (line = white, everything else = black) with `LanesImage.bw_lanes`
5. Blurring the BW masks with `LanesImage.gaussian_blur`
6. Detecting edges with `LanesImage.canny_edge_detection`
7. Creating candidate lane lines with `LanesImage.line_candidates`
8. Smoothing the lines.  Two options are available.
    - First, the candidate lane lines can be merged by applying `LanesImage.smooth_lines` without parameters.  This reduces the candidate lines to one line, stored as `LanesImage.l_avg_line` and `LanesImage.r_avg_line`.  It does so by extending each candidate line to a topline and bottomline for the region of interest.  It then averages the intercepts at these two lines, and returns the line that connects these two intercepts.
    - Second, you may pass in previous lane lines from earlier frames and average them with the current line to achieve some smoothing between frames.  I take advantage of Python's list binding to modify the passed list of previous lines without needing to name a global variable later in the `process_image` function.  To do this averaging with previous frames' lane lines, you can pass a list of previous values with `LanesImage.create_lane_lines`, which will in turn pass these parameters to `LanesImage.smoothlines`.  If these parameters, then `smooth_lines` will first average the current line candidates.  It will then enforce that the # of lines for averaging, including the line from the current frame, is equal to a specified value (currently defaults to 7 frames.)  `smooth_lines` then drops the two outliers for the top line intercepts and bottom line intercepts, and returns the average value for each.
    - This second part has the nice effect of ignoring the lateral jiggle that lane lines sometimes have.  A human driver will typically ignore this jiggle (though it provides a cue for speed) and pay attention to the general location and direction of the lane lines.  So it makes sense to assist "the computer" by removing lateral lane noise from the lane lines.  Averaging over a short period of frames only low risk, as it corresponds to a fraction of a second, and lane keeping is slow moving affair in most situations.
9. Finally, the method `LanesImage.draw_lines` draws the lines on the image.

To use this class one only need create an instance of the class and run the method `create_lane_lines` on the image, which makes for a clean interface.

Each step of the pipeline stores the result as a new class attribute, which is convenient for debugging and can allow additional processing for other features.

I created the class with the idea that one might want to process images differently based on the color of the lane lines or the lighting conditions.  Based on that, I split the image processing into processing the left lane line and the right lane line.  I also developed a function for determining whether a lane line is a white line or yellow line.  However, I only used this feature minimally, using it in step 4 to make BW images of the lines.

The only other changes I made were to tune the various parameter values, like the gaussian kernel size, min_line_length etc.  Probably the trickiest part of this was finding the right values that would consistently yield reasonable lines, but also always pick up lane lines when the lane line was dashed.

To see the pipeline in action, visit the *Debug* section, and uncomment the lines of the form `t.showarray(blah)`, and the step through of the pipeline will be displayed.


###2. Identify potential shortcomings with your current pipeline

I know it has shortcomings because it fails immediately on the challenge video!

So let's discuss how I would approach that problem, and the problem of night time videos.  Essentially this pipeline is relying on very uniform lighting conditions and uniform lane line location and configuration.

One method I considered was to simply record a histogram of edge detections occuring at the top_line of the region of interest (y = 323) and the bottom line (y = 539).  This could capture the sections of that line that are getting 'hit' by the lane line.  Then, connecting the two gives the lane lines.

This might perform poorly on curving lane lines.

A more general version would be to insist that Canny edge detection only yield vertical edges.  Or only vertical and the correct diagonal.  This is part of the original Canny method, so could be an easy implementation.

Another way to improve the algorithm would be to apply a homographic transformation to 'flatten' the road image, and then find the lane lines in that image, which should be vertical or nearly so.

Finally, I'm certain that balancing the image for varying lighting, perhaps equalizing it, or equalizing within regions, is essential for a good result.


###3. Suggest possible improvements to your pipeline

See above!


Thanks,

Alex