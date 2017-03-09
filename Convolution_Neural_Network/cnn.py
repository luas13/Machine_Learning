from yann.network import network
from yann.utils.graph import draw_network

def cnn( dataset= None, verbose=1):

    optimizer_params =  {
                        "momentum_type"       : 'polyak',
                        "momentum_params"     : (0.65, 0.97, 30),
                        "optimizer_type"      : 'rmsprop',
                        "id"                  : "main"
                        }

    dataset_params  = {
                        "dataset"   : dataset,
                        "svm"       : False,
                        "n_classes" : 10,
                        "id"        : 'data'
                        }

    visualizer_params = {
                        "root"       : 'cnn',
                        "frequency"  : 1,
                        "sample_size": 144,
                        "rgb_filters": True,
                        "debug_functions" : False,
                        "debug_layers": False,  # Since we are on steroids this time, print everything.
                        "id"         : 'main'
                        }

    net = network ( borrow = True,
                    verbose = True )

    net.add_module ( type = 'optimizer',
                     params = optimizer_params,
                     verbose = verbose )

    net.add_module ( type = 'datastream',
                     params = dataset_params,
                     verbose = verbose )

    #net.add_module ( type = 'visualizer',
    #                 params = visualizer_params,
    #                 verbose = verbose
    #                )

    # add an input layer
    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose,
                    datastream_origin = 'data',
                    mean_subtract = False )

    # add first convolutional layer
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 30,
                    filter_size = (5,5),
                    pool_size = (2,2),
                    activation = 'relu',
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_1",
                    id = "conv_pool_2",
                    num_neurons = 43,
                    filter_size = (5,5),
                    pool_size = (2,2),
                    activation = 'relu',
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "conv_pool_2",
                    id = "dot_product_1",
                    num_neurons = 1000,
                    dropout_rate = 0.5,
                    activation = ('maxout', 'maxout', 2),
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "dot_product_1",
                    id = "dot_product_2",
                    num_neurons = 1200,
                    dropout_rate = 0.5,
                    activation = ('maxout', 'maxout', 2),
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "dot_product_2",
                    num_classes = 10,
                    regularize = True,
                    activation = 'softmax',
                    verbose = verbose
                    )

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "softmax",
                    objective = "nll",
                    datastream_origin = 'data',
                    regularization = (0.0001, 0.0001),
                    verbose = verbose
                    )

    learning_rates = (0.002, 0.001, 0.0008)
    net.pretty_print()
    #draw_network(net.graph, filename = 'cnn.png')

    net.cook( optimizer = 'main',
              objective_layer = 'obj',
              datastream = 'data',
              classifier_layer = 'softmax',
              verbose = verbose
              )

    net.train( epochs = (8, 5),
               validate_after_epochs = 1,
               training_accuracy = True,
               learning_rates = learning_rates,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)

    net.test(verbose = verbose)

if __name__ == '__main__':
    import sys
    dataset = None
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create_dataset':
            from yann.special.datasets import cook_cifar10
            data = cook_cifar10 (verbose = 2)
            dataset = data.dataset_location()
        else:
            dataset = sys.argv[1]
    else:
        print "provide dataset"

    if dataset is None:
        print " creating a new dataset to run through"
        from yann.special.datasets import cook_cifar10
        # from yann.special.datasets import cook_mnist
        data = cook_cifar10 (verbose = 2)
        # data = cook_mnist()
        dataset = data.dataset_location()

    cnn ( dataset, verbose = 2 )
