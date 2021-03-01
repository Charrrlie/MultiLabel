import warnings


class DefaultConfig(object):
    # data parameters
    data_path = './data/FLICKR-25K.mat'

    training_size = 10000
    query_size = 2000
    database_size = 18015

    load_img_path = './checkpoints/image_model.pth'  
    load_txt_path = './checkpoints/text_model.pth'
    load_cls_path = './checkpoints/cls_model.pth'

    pretrain_model_path = './data/imagenet-vgg-f.mat'
    
    num_class = 24 # label size

    # hyper-parameters
    max_epoch = 500
    batch_size = 128
    
    lr = 1e-4
    
    bit = 16  
    use_gpu = True
    valid = True

    def parse(self, kwargs):
        """
        load configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

