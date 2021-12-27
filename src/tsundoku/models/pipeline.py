import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfTransformer

from tsundoku.features.helpers import to_array
from tsundoku.features.text import score_frequency_table
from tsundoku.helpers import write_json
from tsundoku.models.classifier import PartiallyLabeledXGB, cross_validate
from tsundoku.models.helpers import load_matrix_and_features


def search_tokens(vocabulary, matrix, model_config, token_type):
    group_user_ids = {}
    to_remove_feature_ids = []

    for group, meta in model_config.items():
        try:
            group_tokens = vocabulary[
                vocabulary["token"].isin(meta[token_type]["must_have"])
            ]
        except KeyError:
            print(f"no key: {token_type}")
            continue

        try:
            non_group_tokens = vocabulary[
                vocabulary["token"].isin(meta[token_type]["cant_have"])
            ]
        except KeyError:
            non_group_tokens = None
            print(f"no key: {token_type}")

        group_flag = to_array(matrix[:, group_tokens["token_id"].values].sum(axis=1))
        if non_group_tokens is not None:
            non_group_flag = to_array(
                matrix[:, non_group_tokens["token_id"].values].sum(axis=1)
            )
            group_user_ids[group] = np.where((group_flag > 0) & (non_group_flag == 0))[
                0
            ]
        else:
            group_user_ids[group] = np.where(group_flag > 0)[0]

        to_remove_feature_ids.extend(group_tokens["token_id"].values)

    feature_ids = vocabulary[~vocabulary["token_id"].isin(to_remove_feature_ids)]
    feature_matrix = matrix[:, feature_ids["token_id"].values]

    return feature_matrix, feature_ids, group_user_ids


def update_labels(labels, user_to_row_df, group_user_ids):
    for group, user_idx in group_user_ids.items():
        idx = user_to_row_df[user_to_row_df["row_id"].isin(user_idx)].index.values
        labels.loc[idx, group] = labels.loc[idx, group] + 1

    print(labels.sum(axis=0))
    return labels


def process_matrix(
    path,
    config,
    labels,
    user_to_row_df,
    matrix_key,
    names_key,
    name,
    index="token",
    token_id="token_id",
    tf_idf=False,
):
    raw_matrix, raw_features = load_matrix_and_features(
        path, matrix_key, names_key, name, index=index, token_id=token_id
    )
    matrix, features, labeled_user_ids = search_tokens(
        raw_features, raw_matrix, config, name
    )

    if tf_idf:
        matrix = TfidfTransformer(norm="l1").fit_transform(matrix)

    labels = update_labels(labels.copy(), user_to_row_df, labeled_user_ids)
    return labels, matrix, features, labeled_user_ids


def prepare_features(
    path, config, user_ids, labels, allowed_user_ids=None, tf_idf=False
):
    labels, domain_features, domain_feature_names, domain_labeled_ids = process_matrix(
        path,
        config,
        labels,
        user_ids,
        "user.domains",
        "user.domains",
        "domain",
        index="domain",
    )

    labels, tweet_features, tweet_feature_names, tweet_labeled_ids = process_matrix(
        path,
        config,
        labels,
        user_ids,
        "user.tweets",
        "user.tweet_vocabulary",
        "tweet_tokens",
    )

    (
        labels,
        description_features,
        description_feature_names,
        description_labeled_ids,
    ) = process_matrix(
        path,
        config,
        labels,
        user_ids,
        "user.description_tokens",
        "user.description_tokens",
        "description_token",
    )

    labels, name_features, name_feature_names, name_labeled_ids = process_matrix(
        path,
        config,
        labels,
        user_ids,
        "user.name_tokens",
        "user.name_tokens",
        "user_name",
    )

    (
        labels,
        profile_domain_features,
        profile_domain_feature_names,
        profile_domain_labeled_ids,
    ) = process_matrix(
        path,
        config,
        labels,
        user_ids,
        "user.profile_domains",
        "user.profile_domains",
        "profile_domain",
        index="user.main_domain",
    )

    (
        labels,
        profile_tld_features,
        profile_tld_feature_names,
        profile_tld_labeled_ids,
    ) = process_matrix(
        path,
        config,
        labels,
        user_ids,
        "user.profile_tlds",
        "user.profile_tlds",
        "profile_tld",
        index="user.tld",
    )

    labels, rt_features, rt_feature_names, rt_labeled_ids = process_matrix(
        path,
        config,
        labels,
        user_ids,
        "network.retweet",
        "network.retweet.target_ids.json.gz",
        "rt_target",
        index="index",
        token_id="node_id",
    )

    labels, reply_features, reply_feature_names, reply_labeled_ids = process_matrix(
        path,
        config,
        labels,
        user_ids,
        "network.reply",
        "network.reply.target_ids.json.gz",
        "reply_target",
        index="index",
        token_id="node_id",
    )

    labels, quote_features, quote_feature_names, quote_labeled_ids = process_matrix(
        path,
        config,
        labels,
        user_ids,
        "network.quote",
        "network.quote.target_ids.json.gz",
        "quote_target",
        index="index",
        token_id="node_id",
    )

    # remove contradicting labels
    # keep only labels with one class only
    class_label_counts = labels[labels.values > 0].astype(np.bool).sum(axis=1)
    labels.loc[class_label_counts[class_label_counts > 1].index] = 0

    print(labels.sum(axis=0))

    # add manually annotated accounts from config
    for key, values in config.items():
        if not "account_ids" in values:
            print(f"{key} does not have account ids")
            continue

        tagged_ids = set(values["account_ids"]["known_users"])
        tagged_ids = set(values["account_ids"]["known_users"]) & set(
            user_ids.index.values
        )
        print(f"{key} has {len(tagged_ids)} valid account ids")

        if not tagged_ids:
            continue

        # print(key)
        idx = user_ids.loc[tagged_ids]

        labels.loc[idx.index] = 0
        labels.loc[idx.index, key] = 1

    # remove manually annotated accounts from config
    if allowed_user_ids:
        tagged_ids = set(allowed_user_ids) & set(user_ids.index.values)
        print(
            f"there are {len(tagged_ids)} whitelisted user ids. removing any label from them"
        )

        if tagged_ids:
            idx = user_ids.loc[tagged_ids]
            print("their labels were:")
            print(labels.loc[idx.index].sum())
            labels.loc[idx.index] = 0

    print("done with prelabeling.")
    print(labels.sum(axis=0))

    def network_groups(features):
        rts_per_group = []

        for key in config.keys():
            # this is labeled by user.id, we need row_id
            user_idx = labels[labels[key] > 0].index
            idx = user_ids.loc[user_idx]["row_id"].values
            print(key, user_idx.shape, idx.shape)

            rts_per_group.append(features[:, idx].sum(axis=1))

        rts_per_group = csr_matrix(np.hstack(rts_per_group))
        return rts_per_group

    rts_per_group = network_groups(rt_features)
    replies_per_group = network_groups(reply_features)
    quotes_per_group = network_groups(quote_features)

    feature_names_all = pd.concat(
        [
            # domains
            domain_feature_names[["type", "token"]],
            profile_domain_feature_names[["type", "token"]],
            profile_tld_feature_names[["type", "token"]],
            # tweet dtm
            tweet_feature_names[["type", "token"]],
            # user names
            name_feature_names[["type", "token"]],
            # user description
            description_feature_names[["type", "token"]],
            # user rts
            rt_feature_names[["type", "token"]],
            reply_feature_names[["type", "token"]],
            quote_feature_names[["type", "token"]],
            # rts per group
            pd.DataFrame({"type": "rt_group", "token": config.keys()}),
            pd.DataFrame({"type": "mention_group", "token": config.keys()}),
            pd.DataFrame({"type": "quote_group", "token": config.keys()}),
        ]
    ).reset_index(drop=True)

    X = hstack(
        [
            domain_features,
            profile_domain_features,
            profile_tld_features,
            tweet_features,
            name_features,
            description_features,
            rt_features,
            reply_features,
            quote_features,
            rts_per_group,
            replies_per_group,
            quotes_per_group,
        ],
        format="csr",
    )

    print(X.shape)

    valid_X = X[user_ids.loc[labels.index.values]["row_id"].values, :]

    single_labels = labels.idxmax(axis=1)
    single_labels[labels.sum(axis=1) == 0] = None
    print(single_labels.shape, single_labels.value_counts())

    return valid_X, single_labels, feature_names_all


def evaluate(
    path,
    parameters,
    X,
    labels,
    elem_type,
    stratify_on=None,
    n_splits=5,
    training_eval_fraction=0.1,
):
    outputs = cross_validate(
        parameters,
        X,
        labels.values,
        stratify_on=None,
        n_splits=n_splits,
        eval_fraction=training_eval_fraction,
    )
    write_json(
        outputs, path / f"{elem_type}.classification_model.cross_validation.json.gz"
    )
    return outputs


def train_and_run_classifier(
    parameters,
    X,
    single_labels,
    allowed_user_ids=None,
    eval_fraction=0.15,
    early_stopping_rounds=10,
    threshold_offset_factor=0.1,
    preserve_labels=True,
):
    clf = PartiallyLabeledXGB(xgb_params=parameters)
    clf.fit(
        X,
        single_labels.values,
        eval_fraction=eval_fraction,
        early_stopping_rounds=early_stopping_rounds,
    )

    classes = clf.classes_
    predictions = pd.DataFrame(clf.predict_proba(X), columns=classes)
    predictions["reported_label"] = single_labels.values
    predictions["user.id"] = single_labels.index.values

    predictions["predicted_class"] = predictions[classes].idxmax(axis=1)

    threshold = 1 / len(classes) + threshold_offset_factor
    # apply the threshold
    for key in classes:
        print(key, threshold, predictions[predictions[key] < threshold].shape)
        predictions["predicted_class"][
            (predictions["predicted_class"] == key) & (predictions[key] < threshold)
        ] = "undisclosed"

    print(predictions["predicted_class"].value_counts())

    # enforce labeled users
    if preserve_labels:
        predictions["predicted_class"][
            pd.notnull(single_labels).values
        ] = single_labels[pd.notnull(single_labels)].values
        print(predictions["predicted_class"].value_counts())

    # if whitelisted users were marked as noise, reclassify them as "undisclosed"
    if allowed_user_ids is not None:
        predictions["predicted_class"][
            (predictions["user.id"].isin(allowed_user_ids))
            & (predictions["predicted_class"] == "noise")
        ] = "undisclosed"
        print(predictions["predicted_class"].value_counts())

    print(predictions["predicted_class"].value_counts(normalize=True))

    return clf, predictions


def classifier_pipeline(
    path,
    group_config,
    user_ids,
    labels,
    xgb_parameters,
    allowed_user_ids=None,
    early_stopping_rounds=15,
    eval_fraction=0.15,
    threshold_offset_factor=0.1,
):
    X, single_labels, feature_names_all = prepare_features(
        path, group_config, user_ids, labels, allowed_user_ids=allowed_user_ids
    )
    clf, predictions = train_and_run_classifier(
        xgb_parameters,
        X,
        single_labels,
        allowed_user_ids=allowed_user_ids,
        early_stopping_rounds=early_stopping_rounds,
        eval_fraction=eval_fraction,
        threshold_offset_factor=threshold_offset_factor,
    )
    feature_names_all["xgb.relevance"] = clf.xgb.feature_importances_
    feature_names_all["label"] = (
        feature_names_all["type"].astype(str)
        + ":"
        + feature_names_all["token"].astype(str)
    )

    c_idx = feature_names_all.index.values

    group_vectors = {}

    for idx, group in predictions.groupby("predicted_class"):
        m_idx = group.index.values  # .astype(np.int32)
        print(idx, m_idx.shape)
        group_vectors[idx] = X[m_idx].tocsc()[:, c_idx].sum(axis=0)

    group_vectors = pd.DataFrame(
        np.vstack(group_vectors.values()),
        index=group_vectors.keys(),
        columns=list(feature_names_all["label"].values),
    )

    top_terms = score_frequency_table(group_vectors)

    return clf, predictions, feature_names_all, top_terms, X


def save_classifier(elem_type, path, X, clf, predictions, feature_names_all, top_terms):
    joblib.dump(clf, path / f"{elem_type}.classification_model.joblib.gz")
    save_npz(path / f"{elem_type}.classification.matrix.npz", X)
    feature_names_all.reset_index().to_json(
        path / f"{elem_type}.classification.features.json.gz",
        orient="records",
        lines=True,
    )
    predictions.reset_index().to_json(
        path / f"{elem_type}.classification.predictions.json.gz",
        orient="records",
        lines=True,
    )
    top_terms.reset_index().to_json(
        path / f"{elem_type}.classification.term_associations.json.gz",
        orient="records",
        lines=True,
    )
