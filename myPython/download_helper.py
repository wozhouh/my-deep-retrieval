# -*- coding: utf-8 -*-

# Python function which downloads the cover images by "cmsid"
# usage: get_img(csmid, img_path)

import time
import urllib.request
from urllib.request import urlretrieve


def get_infos_by_cmsid(cmsid):
    url = 'http://inews.webdev.com/cmsGetNewsGeneral?id=%s' % cmsid
    # ret = nameapi.getHostByKey('spider.inews.webdev.com')
    # url = 'http://%s:%d/cmsGetNewsGeneral?id=%s' % (ret[1], ret[2], cmsid)
    # print("url: %s" % url)
    MAX_TRY_TIME = 10
    infos = []
    for try_time in range(MAX_TRY_TIME):
        try:
            req = urllib.request.Request(url)
            urlopen = urllib.request.urlopen(req)
            infos = urlopen.read()
            urlopen.close()
            infos = str(infos.decode('utf-8'))
            break
        except Exception as e:
            print("Exception: try_time=%d, get_infos(), %s, %s" % (try_time+1, cmsid, str(e)))
            if try_time >= MAX_TRY_TIME:
                return False, infos
            time.sleep(0.1)
    return True, infos


def get_general_video_imgurl_from_infos(info):  # for general video image
    info = str(info)  # have \ before /
    key_words = 'newsapp_ls\/0\/'
    pos = info.find(key_words)
    k_n = len(key_words)
    img_url = ''
    if pos != -1:
        pos_begin = pos + k_n
        pos_end_underline = info.find('_', pos_begin)
        pos_end_slash = info.find('\/', pos_begin)
        pos_end = min(pos_end_underline, pos_end_slash)
        new_id = info[pos_begin: pos_end]
        # new_id = info[pos+k_n: pos+k_n+10]  # id length not fixed
        img_url = 'http://inews.gtimg.com/newsapp_ls/0/%s_496280/0' % new_id
    return img_url


def get_img_by_cmsid(csmid, img_path):
    flag, info = get_infos_by_cmsid(csmid)
    if flag:
        img_url = get_general_video_imgurl_from_infos(info)
        urlretrieve(img_url, img_path)
    else:
        print('Fail to get the image of csmid: %s' % csmid)
