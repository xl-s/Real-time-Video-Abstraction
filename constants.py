from plugins import *

# The *_roi constants provide the coordinates of certain key regions.
# As various streamers will have different UI sizes, some of them will
# not be exact; however, all of them (except for centraltext_roi)
# have be tuned to be wide enough to encompass the relevant element
# across a large test sample of streamers.

minimap_roi = [[0.85, 0.74], [1, 1]]    		  # minimap - bottom right position
itemsgold_roi = [[0.55, 0.86], [0.695, 1]]    	  # UI element containing items and gold
gold_roi = [[0.56, 0.965], [0.685, 1]]    		  # region containing only gold count
topright_roi = [[0.8, 0], [1, 0.03]]    		  # information panel on the top right
healthmana_roi = [[0.35, 0.95], [0.575, 1]]    	  # Entire health and mana bar
hpmptext_roi = [[0.43, 0.955], [0.50, 1]]   	  # region containing only HP and MP text
centraltext_roi = [[0.39, 0.12], [0.61, 0.175]]   # may still require adjustment - for the central labels


# The *_lower and *_upper constants provide the colour ranges of certain
# important colours for colour filtering.

hp_lower = [58, 180, 65]    		# HP bar
hp_upper = [68, 255, 255]
mp_lower = [103, 150, 60]			# MP bar
mp_upper = [111, 255, 255]
redbar_lower = [0, 100, 100]		# Non-UI HP bar - red
redbar_upper = [7, 210, 200]
bluebar_lower = [100, 130, 100]    	# Non-UI HP bar - blue
bluebar_upper = [110, 255, 255]
yellowbar_lower = [17, 90, 100]    	# Non-UI HP bar - yellow
yellowbar_upper = [25, 255, 255]
white_lower = [0, 0, 150]    		# General text and minimap bounding box colour
white_upper = [179, 100, 255]    	# may still require adjustment


# Below is a sample of OCR on the HP and MP values.
# First, we work only with the region of image which contains the text
# for HP and MP value. [ ROIFilter(hpmptext_roi) ]
# We then filter out the colours of the HP and MP bars to facilitate
# binarization on the next step. [ hp_colfilter, mp_colfilter ]
# Note that the substitute and invert parameters need to be set to True.
# This substitutes the actual colour of the frame on the mask, and highlights
# the areas which do not match the colour instead of the other way, respectively.
# Then we use Otsu binarization. This should result in the text being
# masked as white, and everything else being black. [ Otsu() ]
# If we do not perform colour filtering in the previous step, it is
# possible for certain parts of the HP/MP bars to be highlighted as well
# in certain cases.
# Since the OCR works best with black text on white background,
# we then invert the image. [ Invert() ]
# Finally, we have the properly pre-processed frame, and can run
# the actual OCR operation on it. [ OCR() ]

hp_colfilter = ColourFilter(hp_lower, hp_upper, substitute=True, invert=True, name='HP')
mp_colfilter = ColourFilter(mp_lower, mp_upper, substitute=True, invert=True, name='MP')

HPMP_OCR_plugins = [ROIFilter(hpmptext_roi), hp_colfilter, mp_colfilter, Otsu(), Invert(), OCR()]