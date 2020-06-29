import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

from .build_network.layer01 import layer_info, load_weights
from .build_network.builder01 import build_network, build_train_op
# import tensorflow as tf
from .training.trainer01 import start_training
from .utils import learningMonitor04
from .prediction.predict02 import reshape_image, mva_findbox, findbox
import numpy as np
import cv2
from .simple_darkflow03 import SimpleDarkflow
from .cython_utils import prediction_utils_cy01 as puc

# from moviepy.editor import VideoFileClip
from mss import mss
from PIL import Image
import time
# from getkey import getkey
import pyxhook

from pynput.mouse import Button, Controller
mouse = Controller()

# Any threads created with the threading module are softoware threads. It's not
# ensured that they are actuall distributed over multiple cores or hardware
# threads. If it is not the case, these threads are executed with time slicing
# and thus there will be no concurrency.
import threading

## GLOBAL VARS ###################
keypress_G = ''
mouse_xy_G = [0, 0]
crop_center_x = int(0.5*600)
crop_center_y = int(0.5*600)
crop_size = 600
find_human = False
shared_frame = np.random.uniform(0,255, size=(600,600,3)).astype(np.uint8)
shared_frame_show = np.random.uniform(0,255, size=(600,600,3)).astype(np.uint8)
CATCH_MONITOR = True
##################################


def OnKeyPress(event):
    global keypress_G
    keypress_G = event.Key
new_hook=pyxhook.HookManager()
new_hook.KeyDown=OnKeyPress
new_hook.HookKeyboard()
new_hook.start()

class Cameron_v2():
    def __init__(self):
        self.sdf = SimpleDarkflow('terrorist_csgo')
        self.meta = dict()
        self.meta['ckpt_folder'] = '/media/salmis10/PrivateData/ExternalDataBase/Data2018_1124_01/Project11/Checkpoints01'
        self.meta['pretrained_weights'] = '/media/salmis10/PrivateData/ExternalDataBase/OpenDataSets/tiny-yolo-voc.weights'
        self.meta['ckpt'] = '2019-01-06_17-00-53'
        self.meta['threshold'] = 0.5
        self.consts = dict()
        self.vars = {'frame_count':0}
        self.time = dict()
        l = 600
        self.consts['l'] = l
        self.mon = {'top':400, 'left':2900 ,'width':l,'height':l}
        # self.vars.update({'crop_center':[int(0.5*l), int(0.5*l)], 'crop_size':int(l)})
        # self.vars.update({'find_human':False})
        self.consts['image_center'] = [int(l/2.), int(l/2.)]
        self.sct = mss()
        threading.Thread(target=send_mouse_sig_thread).start()
        threading.Thread(target=catch_monitor).start()
        time.sleep(2)
        threading.Thread(target=display_th).start()
        self.start()

    def start(self):
        global crop_center_x, crop_center_y, crop_size, find_human
        global shared_frame, CATCH_MONITOR, shared_frame_show
        while True:
            self.time['one_loop_bf'] = time.time()
            self.vars['frame_count'] += 1
            t40 = time.time()
            # self.sct.get_pixels(self.mon)
            # frame = Image.frombytes('RGB', (self.sct.width, self.sct.height), self.sct.image)
            # frame = np.array(frame, order='c')
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = shared_frame
            t41 = time.time()
            self.time['t4'] = t41 - t40
            self.consts['ho'], self.consts['wo'], self.consts['co'] = frame.shape
            l = self.consts['l']
            t00 = time.time()
            # find_human = False
            if find_human :
                dcrop_tlx = int(crop_center_x - 1*crop_size)
                dcrop_brx = int(crop_center_x + 1*crop_size)
                dcrop_tly = int(crop_center_y - 1*crop_size)
                dcrop_bry = int(crop_center_y + 1*crop_size)
                if dcrop_tlx < 0: dcrop_tlx = 0
                if dcrop_tly < 0: dcrop_tly = 0
                frame_dcrop = frame[dcrop_tly:dcrop_bry, dcrop_tlx:dcrop_brx]
            else:
                dcrop_tlx = 0
                dcrop_brx = 600
                dcrop_tly = 0
                dcrop_bry = 600
                frame_dcrop = frame
            t01 = time.time()

            result, B, head = self.sdf.return_predict(self.meta, frame_dcrop)

            # result = self.smooth_box(result)
            tl = (result[0] + dcrop_tlx, result[1] + dcrop_tly)
            br = (result[2] + dcrop_tlx, result[3] + dcrop_tly)
            frame = cv2.rectangle(frame, (dcrop_tlx, dcrop_tly), (dcrop_brx, dcrop_bry), (255,255,0), 1)
            frame = self.add_mark(frame, self.consts['image_center'])
            if result[4] > 0.3:
                if result[4] > 0.6 :
                    frame = cv2.rectangle(frame, tl, br, (0,255,0), 2)
                else:
                    frame = cv2.rectangle(frame, tl, br, (0,255,0), 1)
                hx, hy = head
                if hx != None and hy != None:
                    hx = hx + dcrop_tlx
                    hy = hy + dcrop_tly
                    # hx, hy = self.rescale(hx, hy)
                    # hx, hy = self.remove_outlier(hx, hy)
                    target = (hx, hy+15)
                    self.add_mark(frame, target, 0.01, 'R')
                    if keypress_G == 'x':
                        self.lock_on(self.consts['image_center'], target)
            # self.update_dynamic_crop(result[4], tl, br)
            # print(result)
            # CATCH_MONITOR = True
            t10 = time.time()
            crop_center_x, crop_center_y, crop_size, find_human = \
                puc.update_dynamic_crop(result[4], tl[0], tl[1], br[0], br[1], l)
            t11 = time.time()
            self.time['t1'] = t11 - t10
            # self.vars['crop_center'] = [crop_center_x, crop_center_y]
            # self.vars['crop_size'] = crop_size
            # self.vars['find_human'] = bool(find_human)
            # frame = cv2.putText(frame, t1, (10, 20),
            #                     cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
            # frame = cv2.putText(frame, '{:3d}fps'.format(int(fps)), (80, 20),
            #                     cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            # frame = cv2.putText(frame, 'B=%d'%(B), (180, 20),
            #                     cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            frame = self.say_monitor(frame)
            shared_frame_show = frame

            # cv2.imshow ('frame', frame)
            # if cv2.waitKey(1) & 0xff == ord('q'):
            #     cv2.destroyAllWindows()
            self.time['one_loop_af'] = time.time()
            self.time['t0'] = t01 - t00
            self.say_terminal()




    def say_terminal(self):
        txt = 'fps = %3d, '%(int(1./(self.time['one_loop_af'] - self.time['one_loop_bf'])))
        txt += 'one loop = %2d ms, '%(int(1000*(self.time['one_loop_af'] - self.time['one_loop_bf'])))
        txt += 't0= {:5.4f} ms, '.format(1000*self.time['t0'])
        txt += 't1= {:5.3f} ms, '.format(1000*self.time['t1'])
        txt += 't4= {:5.3f} ms, '.format(1000*self.time['t4'])
        print(txt)

    def say_monitor(self, frame):
        l = self.consts['l']
        if keypress_G == 'x':
            frame = cv2.putText(frame, 'TRACKING', (300-50, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)
        if 'dist' not in self.vars.keys(): return frame
        dist = int(self.vars['dist'])
        frame = cv2.putText(frame, 'dist: %d'%dist, (10, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        if 'sf' not in self.vars.keys(): return frame
        frame = cv2.putText(frame, 'sf: %.3f'%self.vars['sf'], (150, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        return frame

    def smooth_box(self, result):
        if 'w_mva' not in self.vars.keys():
            self.vars['w_mva'] = 0.1
            self.vars['h_mva'] = 0.1
        tlx, tly, brx, bry, prob = result
        w = brx - tlx
        h = bry - tly
        cx = brx - 0.5*w
        cy = bry - 0.5*h
        ratio = 0.7
        thr = 0.1
        epsilon = 1e-3
        if np.abs((w - self.vars['w_mva'])/(self.vars['w_mva'] + epsilon)) < thr:
            self.vars['w_mva'] = self.vars['w_mva'] * ratio + w * (1 - ratio)
        elif self.vars['frame_count'] % 5 == 0:
            self.vars['w_mva'] = self.vars['w_mva'] * ratio + w * (1 - ratio)
        if np.abs((h - self.vars['h_mva'])/(self.vars['h_mva'] + epsilon)) < thr:
            self.vars['h_mva'] = self.vars['h_mva'] * ratio + h * (1 - ratio)
        elif self.vars['frame_count'] % 5 == 0:
            self.vars['h_mva'] = self.vars['h_mva'] * ratio + h * (1 - ratio)
        tlx_new = int((cx - 0.5*self.vars['w_mva']))
        tly_new = int((cy - 0.5*self.vars['h_mva']))
        brx_new = int((cx + 0.5*self.vars['w_mva']))
        bry_new = int((cy + 0.5*self.vars['h_mva']))
        new_result = (tlx_new, tly_new, brx_new, bry_new, prob)
        return new_result

    def add_mark(self, frame, coord, size = 0.05, color='G'):
        x, y = coord
        c = int(frame.shape[0]/2.)
        width = int(frame.shape[0]*size)
        if color == 'G':
            frame[y - width:y + width,x,:] = 0
            frame[y - width:y + width,x,1] = 255
            frame[y, x - width:x + width,:] = 0
            frame[y, x - width:x + width,1] = 255
        if color == 'R':
            frame[y - width:y + width,x,:] = 0
            frame[y - width:y + width,x,2] = 255
            frame[y, x - width:x + width,:] = 0
            frame[y, x - width:x + width,2] = 255
        return frame


    def remove_outlier(self, hx, hy):
        if 'hx_history' not in self.vars.keys():
            self.vars['hx_history'] = [300]*8*2
            self.vars['hy_history'] = [300]*8*2
            self.vars['allow_warp'] = 0
        hx_history = self.vars['hx_history']
        hy_history = self.vars['hy_history']
        dhx = hx_history[-1] - hx_history[-2]
        dhy = hy_history[-1] - hy_history[-2]
        phx = hx_history[-1] + dhx
        phy = hy_history[-1] + dhy
        diff_hx = hx - phx
        diff_hy = hy - phy
        if (diff_hx > dhx * 2) or (diff_hy > dhy * 2):
            if self.vars['allow_warp'] == len(hx_history)-1:
                new_hx = hx
                new_hy = hy
                hx_history = [hx]*len(hx_history)
                hy_history = [hy]*len(hy_history)
            else:
                new_hx = hx_history[1]
                new_hy = hy_history[1]
                self.vars['allow_warp'] += 1
        else:
            new_hx = hx
            new_hy = hy
            hx_history[:-1] = hx_history[1:]
            hx_history[-1] = hx
            hy_history[:-1] = hy_history[1:]
            hy_history[-1] = hy
        self.vars['hx_history'] = hx_history
        self.vars['hy_history'] = hy_history
        return new_hx, new_hy

    def lock_on(self, image_center, target):
        global mouse_xy_G
        if 'diff_history' not in self.vars.keys():
            self.vars['diff_history'] = [np.zeros(2)]*2
        image_center = np.array(image_center)
        target = np.array(target)
        diff = target - image_center # in pixel
        self.vars['diff_history'][:-1] = self.vars['diff_history'][1:]
        self.vars['diff_history'][-1] = diff
        self.auto_trigger(diff)
        self.update_sf()
        mouse_xy_G = self.vars['sf'] * diff

    def update_sf(self):
        if 'sf' not in self.vars.keys():
            self.vars['sf'] = 0.2
            self.vars['dist'] = 0
            self.consts['init_sf'] = 0.2
        diff_history = self.vars['diff_history']
        dist = np.linalg.norm(diff_history[-1])
        self.vars['dist'] = dist
        sf_ub = 0.5
        sf_lb = 0.05
        sf = self.consts['init_sf'] * 0.07 * dist
        sf = min(sf, sf_ub)
        sf = max(sf, sf_lb)
        self.vars['sf'] = sf

    def auto_trigger(self, diff):
        d = np.linalg.norm(diff)
        if d < 12:
            mouse.click(Button.left, 1)


def send_mouse_sig_thread():
    global mouse_xy_G
    f = 40 # Hz
    time2 = time.time()
    while True:
        time.sleep(1./(5*f))
        if (time.time() - time2) > 1./f:
            x, y = mouse_xy_G
            if x != 0 and y != 0:
                mouse.move(x, y)
            time2 = time.time()
            mouse_xy_G = [0, 0]

def display_th():
    global shared_frame_show
    while True:
        frame = shared_frame_show
        cv2.imshow ('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyAllWindows()
        time.sleep(10e-3)

def catch_monitor():
    global shared_frame, CATCH_MONITOR
    fps = 3000
    l = 600
    mon = {'top':400, 'left':2900+1920,'width':l,'height':l}
    sct = mss()
    while True:
        CATCH_MONITOR = True
        if CATCH_MONITOR:
            t0 = time.time()
            sct.get_pixels(mon)
            frame = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            frame = np.array(frame, order='c')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            shared_frame = frame
            # time.sleep(max(1./fps - (time.time() - t0), 0))
            time.sleep(10e-3)
            # print("catch_monitor: {:5.2f} ms".format(1000*(time.time() - t0)))
            # CATCH_MONITOR = False


# self.mon = {'top':400, 'left':2900+1920,'width':l,'height':l}
# # self.vars.update({'crop_center':[int(0.5*l), int(0.5*l)], 'crop_size':int(l)})
# # self.vars.update({'find_human':False})
# self.consts['image_center'] = [int(l/2.), int(l/2.)]
# self.sct = mss()

# cameron = Cameron_v2()
