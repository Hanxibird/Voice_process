class Hyperparams:
    #dataDir = './train_data_list'        # data file


    ########## wave process params ############
    frame_num                 = 320
    sr                        = 48000
    n_fft                     = 2048
    win_length                = 1920
    hop_length                = 480

    nffts                     = n_fft//2 + 1
    nbarks                    = 48
    reciep_filed              = 2 * 3


