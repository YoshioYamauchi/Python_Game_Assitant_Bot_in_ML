
from ..utils.utils02 import generate_traininig_data
from ..utils.utils02 import get_lr
import os
import datetime
import numpy as np
import tensorflow as tf
def start_training(self):
    # test_out = np.random.randn(size=(16, 13, 13, 30))
    # test_image = np.random.uniform(size=(16, 416, 416, 3))
    save_file_name = '/home/salmis/DataBase/OpenDataSets/CSGO/Comp/to06_new'
    test_image_path = '/home/salmis/DataBase/OpenDataSets/CSGO/test_image01.npy'
    test_image = np.load(test_image_path)
    test_fetch = [self.output]
    test_feed_dict = {self.placeholders['input']:test_image}
    test_out = self.sess.run(test_fetch, test_feed_dict)
    np.save(save_file_name, test_out)
    # print test_out
    # merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('../Tensorboard01', self.sess.graph)
    loss_mva = None
    # batches = self.generate_batches() # this is yielder
    batches = generate_traininig_data(self.meta)
    for step, (x_batch, datum) in enumerate(batches):
        feed_dict = {self.placeholders['probs'] : datum['probs'], self.placeholders['confs'] : datum['confs'],
                     self.placeholders['proid'] : datum['proid'], self.placeholders['areas'] : datum['areas'],
                     self.placeholders['coord'] : datum['coord'], self.placeholders['upleft'] : datum['upleft'],
                     self.placeholders['botright'] : datum['botright']}
        # print '987a78', x_batch.shape
        # for key in datum:
        #     filename = '/home/salmis/DataBase/OpenDataSets/CSGO/Comp/p06_'+str(key)
        #     np.save(filename, datum[key])

        feed_dict[self.placeholders['input']] = x_batch # (16, 416, 416, 3)
        # filename = '/home/salmis/DataBase/OpenDataSets/CSGO/Comp/p06_new'
        # np.save(filename, x_batch)
        lr = get_lr(self.meta, step)
        feed_dict[self.placeholders['lr']] = lr
        fetches = [self.train_op, self.loss]
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]
        if loss_mva is None:
            loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        print('Step: %d, loss: %.2f' % (step, loss_mva))
        self.monitor.setPlotByAppend('loss', step, loss)
        self.monitor.setPlotByAppend('lr', step, lr)
        self.monitor.sparseUpdate(10)
        if step % 50 == 0 :
            ckpt_name = os.path.join(self.meta['ckpt_folder'], timestump())
            self.saver.save(self.sess, ckpt_name)

def timestump():
    ts = datetime.datetime.now()
    return '{}'.format( ts.strftime("%Y-%m-%d_%H-%M-%S") )
