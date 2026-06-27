import torch


class Softmax:
    def __init__(self) -> None:
        pass

    def __call__(self, scores: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        scores_max = scores.max()
        scores_exp = (scores - scores_max).exp()
        scores_exp_sum = scores_exp.sum()

        return torch.dot(
            (scores_exp) / scores_exp_sum,
            values,
        )


class StreamingSoftmax:
    def __init__(self):
        # The real unlock is breaking numerator and denominator
        # The exp (dot) values are fused together and tracked
        # The denominator i.e the sum of exp is tracked
        # finally the result is just numerator / denominator

        # This came from a 2018 paper
        # Online normalizer calculation for softmax from Nvidia
        # https://arxiv.org/abs/1805.02867

        self.running_dot = torch.Tensor([0])
        self.running_sum = torch.Tensor([0])
        self.running_max = torch.Tensor([0])

    def update(self, scores: torch.Tensor, values: torch.Tensor) -> None:
        scores_max = scores.max()
        curr_scores_max = torch.max(scores_max, self.running_max)

        # -------------------------------------------------
        # current batch dot and sum
        # -------------------------------------------------
        curr_scores_exp = (scores - curr_scores_max).exp()
        curr_scores_exp_sum = curr_scores_exp.sum()
        curr_dot = torch.dot(curr_scores_exp, values)

        # -------------------------------------------------
        # update running dot and sum
        # -------------------------------------------------
        # Note:
        # self.running_max <= curr_max
        #   case 1: self.running_max < curr_max
        #     running sum and dot are inflated and we need to shrink
        #     so punish by a negative exp
        #   case 2: self.running_max == curr_max
        #     running sum and dot are correct, no change required
        #     so multiply with exp(self.running_max == curr_max) = exp(0) = 1
        shrink_scale = torch.exp(self.running_max - curr_scores_max)

        self.running_sum = (self.running_sum * shrink_scale) + curr_scores_exp_sum
        self.running_dot = (self.running_dot * shrink_scale) + curr_dot
        self.running_max = curr_scores_max

    def finalize(self) -> torch.Tensor:
        return self.running_dot / self.running_sum


def test():
    torch.manual_seed(42)
    size = 10
    batch_size = 2

    s = Softmax()
    scores = torch.rand(size, dtype=torch.float32)
    values = torch.rand(size, dtype=torch.float32)
    softmax_result = s(scores=scores, values=values)

    ss = StreamingSoftmax()
    for offset in range(0, size, batch_size):
        ss.update(
            scores=scores[offset : offset + batch_size],
            values=values[offset : offset + batch_size],
        )
    streaming_softmax_result = ss.finalize()
    print(softmax_result)
    print(streaming_softmax_result)

    torch.testing.assert_close(
        softmax_result,
        streaming_softmax_result[0],
        rtol=1e-6,
        atol=1e-6,
    )


if __name__ == "__main__":
    test()
