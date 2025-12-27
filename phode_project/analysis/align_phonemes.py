import numpy as np

def levenshtein(ref, hyp):
    # Standard dynamic programming
    dp = np.zeros((len(ref)+1, len(hyp)+1))

    for i in range(len(ref)+1):
        dp[i][0] = i
    for j in range(len(hyp)+1):
        dp[0][j] = j

    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )

    return dp
