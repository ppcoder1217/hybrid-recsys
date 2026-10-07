import numpy as np

from cf import SVDRecommender
from content import ContentRecommender


class HybridRecommender:
    def __init__(self, k_user=50, k_item=5, n_factors=50, pop_weight=3.0):
        self.k_user = k_user          # ratings needed for 50% CF trust (user side)
        self.k_item = k_item          # same, for the movie side
        self.n_factors = n_factors
        self.pop_weight = pop_weight

    def fit(self, train, movies):
        self.cf = SVDRecommender(self.n_factors).fit(train)
        self.content = ContentRecommender(self.pop_weight).fit(train, movies)

        # The hybrid scores the FULL catalog (content knows every movie)
        self.items = self.content.items
        self.i_idx = self.content.i_idx

        # Where does each catalog movie sit in the CF model? (-1 = CF never saw it)
        self.cf_col = np.array([self.cf.i_idx.get(m, -1) for m in self.items])
        self.has_cf = self.cf_col >= 0

        self.n_user = train.groupby("user").size().to_dict()
        n_item = train["item"].value_counts().reindex(self.items).fillna(0).values
        self.item_trust = n_item / (n_item + self.k_item)
        return self

    def scores(self, user):
        c = self.content
        # content part: genre similarity + popularity prior
        content = c.pop_weight * c.pop
        if user in c.u_idx:
            content = content + c.X @ c.profiles[c.u_idx[user]]
        content_z = (content - content.mean()) / (content.std() + 1e-9)

        # CF part: dot product of user and movie factors, then z-scored
        cf_z = np.zeros(len(self.items))          # 0 = "no opinion"
        if user in self.cf.u_idx:
            raw = self.cf.item_f @ self.cf.user_f[self.cf.u_idx[user]]
            z = (raw - raw.mean()) / (raw.std() + 1e-9)
            cf_z[self.has_cf] = z[self.cf_col[self.has_cf]]

        # Adaptive weight: more history -> trust CF more
        n = self.n_user.get(user, 0)
        a = (n / (n + self.k_user)) * self.item_trust
        return a * cf_z + (1 - a) * content_z

    def recommend(self, user, seen, k):
        s = self.scores(user)
        seen_idx = [self.i_idx[m] for m in seen if m in self.i_idx]
        s[seen_idx] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        return [self.items[i] for i in top]