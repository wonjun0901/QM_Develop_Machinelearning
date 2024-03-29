from PIL import Image
def image_crop(infilename, save_path):
    """
    image file 와 crop한 이미지를 저장할 path를 입력받아 crop_image를 저장한다.

    """
    
    img = Image.open(infilename)
    (img_h, img_w) = img.size
    print(img.size)

    # crop할 사이즈 : grid_w, grid_h
    grid_w = 20 # crop_width
    grid_h = 20 # crop height
    range_w = (int)(img_w/grid_w)
    range_h = (int)(img_h/grid_h)

    print(range_w, range_h)

    i = 0

    for w in range(range_w):
        for h in range(range_h):
            bbox = (h*grid_h, w*grid_w, (h+1)*(grid_h), (w+1)*(grid_w))
            print(h*grid_h, w*grid_w, (h+1)*(grid_h), (w+1)*(grid_w))
            #가로 세로 시작 가로 세로 끝

            crop_img = img.crop(bbox)

            fname = "{}.jpg".format("{0:05d}".format(i))
            savename = save_path + fname
            
            crop_img.save(savename)

            print('save file' + savename + '....')
            i += 1

if __name__ == '__main__':
    image_crop('C:/Users/wonju/Desktop/#1.tif', 'C:/Users/wonju/Desktop/pre_test/')

