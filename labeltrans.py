Potsdam_list = [[0, 0, 255],
                [138, 0, 255],
                [0, 254, 255],
                [254, 0, 255],
                [254, 254, 255],
                [0, 254, 255]
                ]


def gray2rgb(img_data):
    rgb_img = []
    for line in img_data:
        rgb_line = []
        for data in line:
            rgb_line.append(Potsdam_list[data])
        rgb_img.append(rgb_line)
    return rgb_img
