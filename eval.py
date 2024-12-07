import json
import os
import sys

import matplotlib.pyplot as plt
from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model, conf


# Functions for evaluation


def eval_batch(model, task_sampler, xs, xs_p=None):
    # print("Structure sampling really start")
    task = task_sampler()
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = "cuda"
    else:
        device = "cpu"

    if xs_p is None:
        ys = task.evaluate(xs)
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task.get_metric()(pred.cpu(), ys)
    else:
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
            ys = task.evaluate(xs_comb)

            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

    return metrics


# Functions for generating different kinds of train/test data


def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i: i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
                xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
                xs_test_post_i_orthogonalized
                * xs_test_post_i.norm(dim=2).unsqueeze(2)
                / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i: i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i: i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
        model,
        task_name,
        data_name,
        n_dims,
        n_points,
        prompting_strategy,
        num_eval_examples=1280,
        batch_size=64,
        data_sampler_kwargs={},
        task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    for i in tqdm(range(num_eval_examples // batch_size)):
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)

        metrics = eval_batch(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}

    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs

    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        metrics = {}
        # TODO: changed here, not using linear_regression for plot
        if eval_name != "standard":
            continue
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue
            # if not "gpt" in model.name:
            #     continue
            print(model.name)
            # if "gd" in model.name:
            #     continue
            # TODO: pass data args here
            metrics[model.name] = eval_model(model, **kwargs)
        all_metrics[eval_name] = metrics

    if save_path is not None and len(all_metrics["standard"]) > 0:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
        run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False,
        data_sampler_kwargs={}, task_sampler_kwargs={}, save_path=None
):
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        if torch.cuda.is_available():
            model = model.cuda().eval()
        model = model.eval()
        all_models = [model]
        if not skip_baselines:
            all_models += models.get_relevant_baselines(conf.training.task,
                                                        hidden_layer=task_sampler_kwargs["hidden_layer_size"])
    evaluation_kwargs = build_evals(conf)
    del evaluation_kwargs["linear_regression"]
    evaluation_kwargs["standard"]["data_sampler_kwargs"] = data_sampler_kwargs
    evaluation_kwargs["standard"]["task_sampler_kwargs"] = task_sampler_kwargs

    # if not cache:
    #     save_path = None
    # elif step == -1:
    save_path = os.path.join(run_path, f"{save_path}.json")
    # else:
    #     save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = not skip_model_load
    # recompute = False
    # if save_path is not None and os.path.exists(save_path):
    #     checkpoint_created = os.path.getmtime(run_path)
    #     cache_created = os.path.getmtime(save_path)
    #     if checkpoint_created > cache_created:
    #         recompute = True
    # TODO: pass evaluation_kwargs here, task_name
    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute)
    return all_metrics


def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    assert len(df) == len(df.run_name.unique())
    return df


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')
    from plot_utils import basic_plot, collect_results, relevant_model_names
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_bias', type=float, default=0.)
    parser.add_argument('--data_scale', type=float, default=1.)
    parser.add_argument('--weight_bias', type=float, default=0.)
    parser.add_argument('--weight_scale', type=float, default=1.)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--test', type=str, default="hidden")

    args = parser.parse_args()
    assert args.test in ["data", "weight", "hidden", None]

    run_dir = "models"
    df = read_run_dir(run_dir)

    task = "relu_2nn_regression"

    run_id = "pretrained"  # if you train more models, replace with the run_id from the table above

    run_path = os.path.join(run_dir, task, run_id)
    recompute_metrics = True
    bias_list = [-3, -2, -1, 0, 1, 2, 3]
    scale_list = [1e-2, 1e-1, 1., 2, 5]
    weight_bias_list = [-5, -3, -2, -1, 0, 1, 2, 3, 5]
    weight_scale_list = [1e-2, 1e-1, 1., 2, 5, 10]
    hidden_dim_list = [10, 30, 50, 70, 100, 120, 150, 200, 300, 500]
    data_save_path_fn = lambda _bias, _scale: f"gd_data_b{_bias}_s{_scale}"
    weight_save_path_fn = lambda _bias, _scale: f"gd_weight_b{_bias}_s{_scale}"
    hidden_save_path_fn = lambda _hidden, _: f"gd_hidden_{_hidden}"
    # if args.test == "data":
    #     for data_bias in tqdm(bias_list):
    #         for data_scale in scale_list:
                # print(f"========= {data_scale} =========")
                # data_sampler_params = {"bias": data_bias,
                #                        "scale": data_scale}
                # task_sampler_params = {"weight_bias": args.weight_bias,
                #                        "weight_scale": args.weight_scale,
                #                        "hidden_layer_size": args.hidden_size}
                # save_path_name = data_save_path_fn(data_bias, data_scale)
                # if recompute_metrics:
                #     get_run_metrics(run_path, cache=True, data_sampler_kwargs=data_sampler_params,
                #                     task_sampler_kwargs=task_sampler_params, save_path=save_path_name)
    # if args.test == "weight":
    #     for weight_bias in tqdm(weight_bias_list):
    #         for weight_scale in weight_scale_list:
    #             print(f"========= {weight_scale} =========")
    # data_bias = 0.
    # data_scale = 4.
    #     data_sampler_params = {"bias": args.data_bias,
    #                            "scale": args.data_scale}
    #     task_sampler_params = {"weight_bias": weight_bias,
    #                            "weight_scale": weight_scale,
    #                            "hidden_layer_size": args.hidden_size}
    #     save_path_name = weight_save_path_fn(weight_bias, weight_scale)
    #     if recompute_metrics:
    #         get_run_metrics(run_path, cache=True, data_sampler_kwargs=data_sampler_params,
    #                         task_sampler_kwargs=task_sampler_params, save_path=save_path_name)

    # if args.test == "hidden":
    #     # for hidden_dim in tqdm(hidden_dim_list):
    hidden_dim = 10
    #     data_sampler_params = {"bias": args.data_bias,
    #                            "scale": args.data_scale}
    #     task_sampler_params = {"weight_bias": args.weight_bias,
    #                            "weight_scale": args.weight_scale,
    #                            "hidden_layer_size": hidden_dim}
    #     save_path_name = hidden_save_path_fn(hidden_dim, None)
    #     if recompute_metrics:
    #         get_run_metrics(run_path, cache=True, data_sampler_kwargs=data_sampler_params,
    #                         task_sampler_kwargs=task_sampler_params, save_path=save_path_name)
    # exit()
    # if not args.test is None:
    #     exit()
    def valid_row(r):
        return r.task == task and r.run_id == run_id
    #
    kwd = "gd_hidden"
    # name_fn = data_save_path_fn
    # name_fn = weight_save_path_fn
    name_fn = hidden_save_path_fn
    # result_dict = {}
    # # for hidden_dim in hidden_dim_list:
    params = (hidden_dim, None)
    # # for weight_bias in weight_bias_list:
    # #     for weight_scale in weight_scale_list:
    # params = (weight_bias, weight_scale)
    # for data_bias in bias_list:
    #     for data_scale in scale_list:
    # params = (data_bias, data_scale)
    data_save_path = name_fn(*params)
    metrics = collect_results(run_dir, df, valid_row=valid_row, save_path=data_save_path)
    #         result_dict[params] = metrics["standard"]["Transformer"]

    _, conf = get_model_from_run(run_path, only_conf=True)
    n_dims = conf.model.n_dims
    # (-3, 5), (-2, 5), (-1, 5), (0, 5), (N, 5)
    models = relevant_model_names[task]
    fig, ax = basic_plot(metrics["standard"], models=models)
    # fig, ax = basic_plot(result_dict, models=models)

    fig.savefig(f"{kwd}.png", dpi=300)

