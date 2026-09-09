from geomloss import SamplesLoss

sinkhorn = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur=0.05,
)
