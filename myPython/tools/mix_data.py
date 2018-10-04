# -*- coding: utf-8 -*-

import os

if __name__ == '__main__':
    paris_dataset = '/home/processyuan/data/Paris/cls/'  # 6412 images in total
    cover_dataset = '/home/processyuan/NetworkOptimization/cover/training/img/'
    dst_dataset = '/home/processyuan/data/Paris/test/img/'
    paris_black_list = ['paris_louvre_000146.jpg',
                        'paris_triomphe_000867.jpg',
                        'paris_moulinrouge_000422.jpg',
                        'paris_museedorsay_001059.jpg',
                        'paris_sacrecoeur_000299.jpg',
                        'paris_notredame_000188.jpg',
                        'paris_pompidou_000196.jpg',
                        'paris_triomphe_000662.jpg',
                        'paris_pantheon_000960.jpg',
                        'paris_pompidou_000467.jpg',
                        'paris_pompidou_000201.jpg',
                        'paris_pantheon_000284.jpg',
                        'paris_louvre_000136.jpg',
                        'paris_pantheon_000974.jpg',
                        'paris_sacrecoeur_000330.jpg',
                        'paris_triomphe_000863.jpg',
                        'paris_triomphe_000833.jpg',
                        'paris_pompidou_000195.jpg',
                        'paris_sacrecoeur_000353.jpg',
                        'paris_pompidou_000640.jpg',
                        ]
    img_paris = 6412
    img_cover = 6000
    img_num = img_paris + img_cover
    cnt = 0  # for checking
    # img_list = []

    # Paris
    img_list_paris = os.listdir(paris_dataset)
    for cls in img_list_paris:
        cls_path = os.path.join(paris_dataset, cls)
        img_list_cls = os.listdir(cls_path)
        for img in img_list_cls:
            if img not in paris_black_list:
                cnt += 1
                open(os.path.join(dst_dataset, img), 'wb').write(open(os.path.join(cls_path, img), 'rb').read())
        # img_list.extend(img_list_cls)

    # cover
    img_list_cover = os.listdir(cover_dataset)
    for img in img_list_cover[: img_cover]:
        cnt += 1
        open(os.path.join(dst_dataset, img), 'wb').write(open(os.path.join(cover_dataset, img), 'rb').read())
    # img_list.extend(img_list_cover)

    print("finish the dataset of %d images" % cnt)
