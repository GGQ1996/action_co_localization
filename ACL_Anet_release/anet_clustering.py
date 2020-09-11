import os
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import json
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist
import scipy


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


anet_label_path = "activity_net.v1-2.min-missing-removed.json"
flow_path = "ANET_I3D_FEATURE//flow-resize-step16"
rgb_path = "ANET_I3D_FEATURE//rgb-resize-step16"
atten_weight_path = "your_attention_weight_path"
anet_label = load_json(anet_label_path)
anet_label = anet_label['database']

all_flow_file = os.listdir(flow_path)
all_rgb_file = os.listdir(rgb_path)


def get_train_label(anet_label):
    training_index = []
    action_2_video = {}
    video_2_action = {}
    for tv in anet_label:
        if anet_label[tv]["subset"] != "training":
            continue
        training_index.append(tv)
        tc = anet_label[tv]["annotations"][0]["label"]
        if action_2_video.has_key(tc):
            action_2_video[tc].append(tv)
        else:
            action_2_video[tc] = []
            action_2_video[tc].append(tv)

        if video_2_action.has_key(tv):
            video_2_action[tv].append(tc)
        else:
            video_2_action[tv] = []
            video_2_action[tv].append(tc)

    for tk in action_2_video:
        action_2_video[tk] = list(set(action_2_video[tk]))
    for tk in video_2_action:
        video_2_action[tk] = list(set(video_2_action[tk]))
    return training_index, action_2_video, video_2_action


def get_feature(training_index, flow_path, rgb_path, atten_weight_path=None):
    if atten_weight_path is None:
        print("Avg pooling feature")
    else:
        print("Attention pooling feature")
    all_atten_fea = []
    late_fusion_flag = False
    global_score_flag = False
    cnt = 0
    for tv_name in training_index:
        cnt += 1
        if cnt % 500 == 0:
            print("%d loaded" % cnt)
        tf = "v_" + tv_name + "-flow.npz"
        sig_flow_path = os.path.join(flow_path, tf)
        flow_data = np.load(sig_flow_path)
        flow_fea = flow_data['feature']

        tr = "v_" + tv_name + "-rgb.npz"
        sig_rgb_path = os.path.join(rgb_path, tr)
        rgb_data = np.load(sig_rgb_path)
        rgb_fea = rgb_data['feature']

        # avg fea
        if atten_weight_path is None:
            att_weight_flow_fea = np.mean(flow_fea[0, :, :], axis=0)
            att_weight_rgb_fea = np.mean(rgb_fea[0, :, :], axis=0)

        elif late_fusion_flag:
            tcas = tv_name + ".npz"
            cas = np.load(os.path.join(atten_weight_path, tcas))
            att_weight = cas['weight']

            # downsample
            tmp_flow_fea = flow_fea[0, :, :]
            tmp_rgb_fea = rgb_fea[0, :, :]

            atten_flow_fea = tmp_flow_fea * att_weight
            att_weight_flow_fea = np.sum(atten_flow_fea, axis=0)
            atten_rgb_fea = tmp_rgb_fea * att_weight
            att_weight_rgb_fea = np.sum(atten_rgb_fea, axis=0)

            if global_score_flag is True:
                global_score = cas['global_score']
                global_score = global_score / np.linalg.norm(global_score)

        else:
            # print("use diff weight for rgb and flow")
            tcas = tv_name + ".npz"
            rgb_weight_path = atten_weight_path.replace("both", "rgb")
            flow_weight_path = atten_weight_path.replace("both", "flow")

            rgb_cas = np.load(os.path.join(rgb_weight_path, tcas))
            flow_cas = np.load(os.path.join(flow_weight_path, tcas))

            rgb_att_weight = rgb_cas['weight']
            flow_att_weight = flow_cas['weight']

            # downsample
            tmp_flow_fea = flow_fea[0, :, :]
            tmp_rgb_fea = rgb_fea[0, :, :]

            atten_flow_fea = tmp_flow_fea * flow_att_weight
            att_weight_flow_fea = np.sum(atten_flow_fea, axis=0)
            atten_rgb_fea = tmp_rgb_fea * rgb_att_weight
            att_weight_rgb_fea = np.sum(atten_rgb_fea, axis=0)

            if global_score_flag is True:
                rgb_global = rgb_cas['global_score']
                flow_global = flow_cas['global_score']
                global_score = (rgb_global + flow_global) / 2.0
                global_score = global_score / np.linalg.norm(global_score)

        # normalize flow and rgb seperately
        att_weight_flow_fea = att_weight_flow_fea / np.linalg.norm(att_weight_flow_fea)
        att_weight_rgb_fea = att_weight_rgb_fea / np.linalg.norm(att_weight_rgb_fea)

        if (atten_weight_path is None) or (global_score_flag is False):
            fuse_fea = np.concatenate((att_weight_rgb_fea, att_weight_flow_fea), axis=0)
        elif global_score_flag:
            fuse_fea = np.concatenate((att_weight_rgb_fea, att_weight_flow_fea, global_score), axis=0)
        all_atten_fea.append(fuse_fea)
    return all_atten_fea


def get_subset(num_subset_class, action_2_video, training_index, all_atten_fea):
    # num_subset_cluster = num_subset_class
    subset_class = action_2_video.keys()[0:num_subset_class]
    subset_index = []
    subset_atten_fea = []
    for tc in subset_class:
        tc_sub = action_2_video[tc]
        for tv in tc_sub:
            subset_index.append(tv)
            tv_fea_index = training_index.index(tv)
            subset_atten_fea.append(all_atten_fea[tv_fea_index])
    subset_atten_fea = np.stack(subset_atten_fea, axis=0)
    return subset_class, subset_index, subset_atten_fea


def get_affinity(video_index, video_feature, action_2_video):
    sorted_video_index = []
    sorted_video_fea = []
    cnt = 0
    for tmp_act in action_2_video:
        for tmp_vid in action_2_video[tmp_act]:
            if tmp_vid in sorted_video_index:
                continue
            sorted_video_index.append(tmp_vid)
            tmp_vid_index = video_index.index(tmp_vid)
            sorted_video_fea.append(video_feature[tmp_vid_index])

    num_video = len(sorted_video_index)
    weight = np.zeros((num_video, num_video))
    beta = 0
    cnt = 0
    for i in range(num_video):

        for j in range(i, num_video):
            # calculate gamma
            cnt += 1
            beta += np.linalg.norm(sorted_video_fea[i] - sorted_video_fea[j])

            dis = np.square(np.linalg.norm(sorted_video_fea[i] - sorted_video_fea[j]))
            weight[i][j] = dis
            weight[j][i] = dis

    beta = beta / cnt
    gamma = - 1.0 / (2 * beta * beta)
    print("gamma is %f " % gamma)

    weight = np.exp(gamma * weight)

    return weight, sorted_video_index, sorted_video_fea


training_index, action_2_video, video_2_action = get_train_label(anet_label)
att_fea_v1 = get_feature(training_index, flow_path, rgb_path, atten_weight_path=atten_weight_path)


def get_cluster_performance(num_of_cluster, label_pred, subset_index, action_2_video, video_2_action):
    cluster_res = {}
    for tmp_cls in range(num_of_cluster):
        cluster_res[tmp_cls] = []
    for i in range(len(label_pred)):
        tmp_cls = label_pred[i]
        file_name = subset_index[i]
        cluster_res[tmp_cls].append(file_name)

    all_precision = []
    all_recall = []
    all_cluster_label = []
    # add cluster index to action class mapping
    cluser_2_action = {}
    soft_cluster_2_action = {}

    total_true_cnt = 0

    # all class
    for label_index in range(num_of_cluster):
        all_class = list(action_2_video.keys())
        # print(all_class)
        action_cnt = {}
        for tc in all_class:
            action_cnt[tc] = 0
        # print(label_index)
        for tv in cluster_res[label_index]:
            tv_label = video_2_action[tv]
            for sig_label in tv_label:
                action_cnt[sig_label] += 1

            # set the label of cluster as the class which appear most
            max_cnt = 0
            cluster_label = ''
            for tmp_label in action_cnt:
                if action_cnt[tmp_label] > max_cnt:
                    max_cnt = action_cnt[tmp_label]
                    cluster_label = tmp_label
        all_cluster_label.append(cluster_label)

        # add cluster index to action class mapping
        cluser_2_action[label_index] = cluster_label

        soft_cluster_2_action[label_index] = []
        cluster_video_num = len(cluster_res[label_index])

        for tmp_label in action_cnt:
            if action_cnt[tmp_label] == 0:
                continue
            if action_cnt[tmp_label] == (max_cnt):
                # tmp_label_weight = 1.0 * action_cnt[tmp_label] / cluster_video_num
                tmp_label_weight = 1.0
                soft_cluster_2_action[label_index].append([tmp_label, tmp_label_weight])
            elif action_cnt[tmp_label] >= 0.5 * max_cnt:
                tmp_label_weight = 0.5
                soft_cluster_2_action[label_index].append([tmp_label, tmp_label_weight])

        precision = 1.0 * max_cnt / len(cluster_res[label_index])
        recall = 1.0 * max_cnt / len(action_2_video[cluster_label])
        total_true_cnt += max_cnt

        print("********")
        print("cluster label %d" % label_index)
        print("num of video in cluster %d" % (len(cluster_res[label_index])))
        print("match class %s" % (cluster_label))
        print("num of all gt video %d" % (len(action_2_video[cluster_label])))
        print("num of cluster gt video %d" % (max_cnt))
        print("precision %.4f" % precision)
        print("recall %.4f\n" % recall)
        action_cnt = sorted(action_cnt.items(), key=lambda e: e[1], reverse=True)
        print(action_cnt[0:10])
        print("\n")
        # print(soft_cluster_2_action[label_index])

        all_precision.append(precision)
        all_recall.append(recall)

    average_prec = np.mean(np.array(all_precision))
    average_recall = np.mean(np.array(all_recall))
    print("avg prec")
    print(average_prec)
    print("avg recall")
    print(average_recall)
    print("all prec %.4f" % (1.0 * total_true_cnt / len(subset_index)))

    all_cluster_label = list(set(all_cluster_label))
    print("num of cluster label %d" % (len(all_cluster_label)))

    video_index = subset_index
    gt_label = []
    for tv in video_index:
        gt_label.append(video_2_action[tv])
    #     non_over_label = []
    #     for i in range(label_pred.shape[0]):
    #         non_over_label.append(int(label_pred[i]))
    gt_label = np.array(gt_label)
    gt_label = np.squeeze(gt_label)
    # print(gt_label.shape)
    from sklearn import metrics
    print("Adjusted rand score %.4f" % metrics.adjusted_rand_score(gt_label, label_pred))
    print("NMI %.4f" % metrics.normalized_mutual_info_score(gt_label, label_pred))

    return cluser_2_action, soft_cluster_2_action


num_subset_class = 100
subset_class, subset_index, subset_atten_fea = get_subset(num_subset_class, action_2_video, training_index, att_fea_v1)

affinity_matrix, sorted_video_index, sorted_video_fea = get_affinity(subset_index, subset_atten_fea, action_2_video)
subset_atten_fea = sorted_video_fea
subset_index = sorted_video_index

# cluster
num_of_cluster = num_subset_class
estimator = SpectralClustering(n_clusters=num_of_cluster, random_state=0, affinity='precomputed')
estimator.fit_predict(affinity_matrix)
label_pred = estimator.labels_
cluser_2_action, soft_cluster_2_action = get_cluster_performance(num_of_cluster, label_pred, subset_index,
                                                                 action_2_video, video_2_action)
