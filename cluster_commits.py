import openai
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from git import Repo
import numpy as np
import os
import subprocess

# Repo_path is path to repo to collect commits, max_num_commits maximum number
# of commits to put into num_clusters clusters
def GetCommitClusters(
    repo_path: str,
    git_log_cmd: str,
    max_num_commits: int,
    num_clusters: int
    ):

    # Initialize a Git repository object with GitPython
    repo = Repo(repo_path)

    # Define the Bash command to get the commits
    cmd=f'cd {repo_path} && '
    cmd+=git_log_cmd

    # Run the Bash command and capture its output
    completed_process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)

    # Split the output into a list using newline as the delimiter
    output_list = completed_process.stdout.split('\n')

    commit_messages = []

    # Store the commit messages list
    i = 0
    for commit_hash in output_list:
        commit = repo.commit(commit_hash)
        message = commit.message.strip()
        if "Merge" not in message:
            commit_messages.append(message)
            i += 1
            if i > max_num_commits:
                break

    # Use TF-IDF Vectorizer to convert commit messages to numerical vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(commit_messages)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)

    # get cluster labels for each commit
    cluster_labels = kmeans.labels_

    # cluster the commits
    commit_groups = {}
    for i, label in enumerate(cluster_labels):
        if label not in commit_groups:
            commit_groups[label] = []
        commit_groups[label].append(commit_messages[i])

    return commit_groups.items()

# give this function cluster items and create a single commit message
def GenCommitMessage(cluster_items, api_key):
    # set opeanai key and prompt
    openai.api_key = api_key
    prompt = "Please create a commit message that summarizes the following \
list of commits:\n"

    # print commits in each cluster to a long commit for that cluster
    long_commits = []
    i = 0
    for label, commits in cluster_items:
        #print(f"Cluster {label}:")
        long_commits.append("")
        for commit in commits:
            long_commits[i] += commit + "\n"

        # Generate a response using the prompt
        response = openai.Completion.create(
            # change perhaps to gpt 3.5 turbo? need to investigate
            engine="text-davinci-003",
            prompt=prompt+long_commits[i],
        )

        # Extract the generated description
        single_message = response.choices[0].text.strip()

        print(f"Cluster {label}\n all commits in cluster:\n {long_commits[i]}\n \
single commit message: {single_message}")
        i += 1

cmd = "git log --pretty=format:\"%h\""

cluster_items = GetCommitClusters(repo_path, cmd, 300, 20)

GenCommitMessage(cluster_items, api_key)
