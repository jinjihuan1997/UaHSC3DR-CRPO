import argparse
from macarons.testers.magician_planning import *

dir_path = os.path.abspath(os.path.dirname(__file__))
test_configs_dir = os.path.join(dir_path, "./configs/test/")


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Script to test a full macarons model in large 3D scenes.')
    parser.add_argument('-c', '--config', type=str, help='name of the config file. '
                                                         'Default is "test_in_default_scenes_config.json".')
    parser.add_argument('--r_aux', type=float, default=1.0, help='Ratio of valid depth pixels kept for planning.')
    parser.add_argument('--degrade_mode', type=str, default='block',
                        choices=('block', 'pixel_dropout', 'downsample_upsample'),
                        help='Planning-level depth degradation mode. '
                             '"block": block-wise dropout. '
                             '"pixel_dropout": uniform random pixel dropout. '
                             '"downsample_upsample": bilinear downsample+nearest upsample '
                             '(r_aux is the linear scale factor).')
    parser.add_argument('--degrade_seed', type=int, default=0, help='Base seed for depth degradation.')
    parser.add_argument('--degrade_block_size', type=int, default=16, help='Block size for block depth degradation.')

    args = parser.parse_args()

    if args.config:
        params_name = args.config
    else:
        params_name = "test_in_default_scenes_config.json"

    params_name = os.path.join(test_configs_dir, params_name)
    test_params = load_params(params_name)
    test_params.r_aux = args.r_aux
    test_params.degrade_mode = args.degrade_mode
    test_params.degrade_seed = args.degrade_seed
    test_params.degrade_block_size = args.degrade_block_size


    with torch.no_grad():
        run_magician_test(params_name=test_params.params_name,
                 model_name=test_params.model_name,
                 results_json_name=test_params.results_json_name,
                 numGPU=test_params.numGPU,
                 test_scenes=test_params.test_scenes,
                 test_resolution=test_params.test_resolution,
                 use_perfect_depth_map=test_params.use_perfect_depth_map,
                 compute_collision=test_params.compute_collision,
                 load_json=test_params.load_json,
                 dataset_path=test_params.dataset_path,
                 test_params=test_params)
