import jax
from jax import numpy as jnp
from flax import nnx
import optax
from tqdm import tqdm
import numpy as np

from data import get_dataloaders
from model import ResNet

from jax.sharding import Mesh, PartitionSpec, NamedSharding

from torch.utils.tensorboard import SummaryWriter


def compute_metrics(loss, logits_bC, y_b):
    softmaxed_logits_bC = jax.nn.softmax(logits_bC)
    top1_acc = jnp.mean(jnp.argmax(softmaxed_logits_bC, -1) == y_b)

    _, top5_preds = jax.lax.top_k(softmaxed_logits_bC, 5)
    top5_acc = jnp.mean(jnp.isin(y_b, top5_preds))

    metrics = {
        "loss": loss,
        "top1-accuracy": top1_acc,
        "top5-accuracy": top5_acc,
    }

    return metrics


def loss_fn(model, x_bhwc, y_b):
    logits_bC = model(x_bhwc)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits_bC, y_b))
    return loss, logits_bC


def strip_padded_samples(x_bhwc, y_b):
    n_non_pad = jnp.where(y_b == -1, 0, 1).sum()
    updated_x_bhwc = jax.lax.dynamic_slice(x_bhwc, (0, 0, 0, 0), (n_non_pad, *x_bhwc.shape[1:]))
    updated_y_b = jax.lax.dynamic_slice(y_b, (0,), (n_non_pad,))
    return updated_x_bhwc, updated_y_b


@nnx.jit
def train_step(model, optimizer, x_bhwc, y_b):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits_bC), grads = grad_fn(model, x_bhwc, y_b)

    optimizer.update(grads)

    return compute_metrics(loss, logits_bC, y_b)


def train_epoch(model, optimizer, mesh, per_device_batch_size, train_dataloader, epoch_num, writer, batch_counter):
    epoch_losses, epoch_top1accuracies, epoch_top5accuracies = [], [], []
    image_data_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None))
    label_data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    with mesh:
        for x_bhwc, y_b in tqdm(train_dataloader, desc=f"Training loop: {epoch_num}"):
            x_bhwc, y_b = x_bhwc.numpy(), y_b.numpy()

            # pad batch if necessary
            n = y_b.shape[0]
            # skip batch if not divisible by number of devices. not ideal but c'est la vie
            if n % per_device_batch_size > 0:
                break

            # shard the data
            x_bhwc = jax.device_put(x_bhwc, image_data_sharding)
            y_b = jax.device_put(y_b, label_data_sharding)

            batch_metrics = train_step(model, optimizer, x_bhwc, y_b)
            print(batch_metrics)
            epoch_losses.append(batch_metrics["loss"])
            epoch_top1accuracies.append(batch_metrics["top1-accuracy"])
            epoch_top5accuracies.append(batch_metrics["top5-accuracy"])
            writer.add_scalar(
                "Batch Top1 Accuracy/train",
                float(batch_metrics["top1-accuracy"]),
                batch_counter + 1,
            )
            writer.add_scalar(
                "Batch Top5 Accuracy/train",
                float(batch_metrics["top5-accuracy"]),
                batch_counter + 1,
            )
            writer.add_scalar(
                "Batch Loss/train", float(batch_metrics["loss"]), batch_counter + 1
            )
            batch_counter += 1

    return (
        np.mean(epoch_losses),
        np.mean(epoch_top1accuracies),
        np.mean(epoch_top5accuracies),
        batch_counter,
    )


def train(model, train_dataloader, num_epochs: int, per_device_batch_size: int, train_dataloader_len: int):
    boundaries_and_scales = {20 * train_dataloader_len: 0.1, 30 * train_dataloader_len: 0.1, 40 * train_dataloader_len: 0.1}
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=0.1, boundaries_and_scales=boundaries_and_scales
    )
    momentum = 0.9
    optimizer = nnx.Optimizer(model, optax.sgd(lr_schedule, momentum))
    mesh = Mesh(
        devices=np.array(jax.devices()).reshape(
            4,
        ),
        axis_names=("data"),
    )


    batch_counter = 0
    for epoch in tqdm(range(num_epochs), desc=f"Main training loop"):
        avg_loss, avg_top1acc, avg_top5acc, batch_counter = train_epoch(
            model, optimizer, mesh, per_device_batch_size, train_dataloader, epoch, writer, batch_counter
        )
        print(f"Avg stats: {avg_loss, avg_top1acc, avg_top5acc}")
        writer.add_scalar("Epoch Top1 Accuracy/train", float(avg_top1acc), epoch + 1)
        writer.add_scalar("Epoch Top5 Accuracy/train", float(avg_top5acc), epoch + 1)
        writer.add_scalar("Epoch Loss/train", float(avg_loss), epoch + 1)

@nnx.jit
def create_sharded_model():
    model = ResNet(num_layers=34, rngs=nnx.Rngs(params=0))  # Unsharded at this moment.
    state = nnx.state(model)  # The model's state, a pure pytree.
    pspecs = nnx.get_partition_spec(state)  # Strip out the annotations from state.
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)  # The model is sharded now!
    return model


if __name__ == "__main__":
    num_epochs = 45
    batch_size = 256
    writer = SummaryWriter(f"./logs/sgd_{num_epochs}_{batch_size}")

    train_dataloader, val_dataloader = get_dataloaders(batch_size)
    mesh = Mesh(
        devices=np.array(jax.devices()).reshape(
            4,
        ),
        axis_names=("data"),
    )
    per_device_batch_size = batch_size // 4
    with mesh:
        sharded_model = create_sharded_model()
    # print(len(train_dataloader))
    train(sharded_model, train_dataloader, num_epochs, per_device_batch_size, len(train_dataloader))
