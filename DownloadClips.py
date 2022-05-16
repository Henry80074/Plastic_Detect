import ee
import multiprocessing
import os
import requests
import shutil
from retry import retry

ee.Initialize(opt_url='https://earthengine.googleapis.com')

StartDate= ee.Date("2021-01-01")
EndDate = ee.Date("2021-02-01")

region2 = ee.Geometry.Polygon([[-86.243, 17.073],[-88.956, 17.073], [-88.956, 15.671], [-86.243, 15.671]], None, False)

image = (ee.ImageCollection('COPERNICUS/S2_SR')
         .filterDate(StartDate, EndDate)
         .filterBounds(region2)
         .mosaic()
         .clip(region2)
         .select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']))

# image = ee.ImageCollection('COPERNICUS/S2_SR') \
#             .filterBounds(region2) \
#             .filterDate(StartDate, EndDate) \
#             .select('B4', 'B3', 'B2') \
#             .median() \
#             .visualize(min=0, max=1000) \
#             .clip(region2)


params = {
    'buffer': 1280,  # The buffer distance (m) around each point
    'scale': 2560,  # The scale to do stratified sampling
    'seed': 1,  # A randomization seed to use for subsampling.
    'dimensions': '256x256',  # The dimension of each tile - match highest resolution of band e.g 1 pixel per 10 meters
    'format': "GEO_TIFF",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
    'prefix': 'tile_',  # The filename prefix
    'processes': 25,  # How many processes to used for parallel processing
    'out_dir': './tiles',  # The output directory. Default to the current working directly
}
# viz_params = {
#   'bands': ['B8', 'B4', 'B3'], 'min':0, 'max': 4000, 'gamma': 1
# }

def getRequests():
    img = ee.Image(1).rename("Class").addBands(image)
    # gets grid of uniform points for downloading each tile of the image collection
    # scale should be size of tile in meters(squared)
    points = img.sampleRegions(collection=ee.FeatureCollection([region2]), scale=params['scale'], geometries=True)
    return points.aggregate_array('.geo').getInfo()

@retry(tries=10, delay=1, backoff=2)
def getResult(index, point):
    print("running..")
    point = ee.Geometry.Point(point['coordinates'])
    region = point.buffer(params['buffer']).bounds()
    if params['format'] in ['png', 'jpg']:
        url = image.getThumbURL(
            {
                'region': region,
                'dimensions': params['dimensions'],
                'format': params['format'],
            }
        )
    else:
        url = image.getDownloadURL(
            {
                'region': region,
                'dimensions': params['dimensions'],
                'format': params['format'],
            }
        )

    if params['format'] == "GEO_TIFF":
        ext = 'tif'
    else:
        ext = params['format']

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    out_dir = os.path.abspath(params['out_dir'])
    basename = str(index)
    filename = f"{params['prefix']}{basename}.{ext}"
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", basename)

items = getRequests()
if __name__ == '__main__':
    pool = multiprocessing.Pool(25)
    pool.starmap(getResult, enumerate(items))
    pool.close()
    pool.join()