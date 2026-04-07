import math
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from hmotiongpt.data.collators import SFTCollator
from hmotiongpt.training.common import (
    build_dataset,
    create_projector,
    create_run_dir,
    create_scheduler,
    create_tokenizer_and_model,
    create_writer,
    prepare_runtime,
    resolve_optional_path,
    safe_ppl,
    save_epoch_metrics,
    save_run_config,
)


def run_sft(config: Dict[str, Any]) -> Path:
    device = prepare_runtime(config)
    dataset = build_dataset(config)
    tokenizer, imu_llm, imu_token_ids = create_tokenizer_and_model(config, freeze_llm=False)
    d_model = imu_llm.llm.get_input_embeddings().embedding_dim
    projector = create_projector(config, d_model=d_model)
    projector_path = resolve_optional_path(config, config["model"]["projector_path"])
    projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
    projector.eval()
    for parameter in projector.parameters():
        parameter.requires_grad = False
    collator = SFTCollator(
        tokenizer=tokenizer,
        imu_token_ids=imu_token_ids,
        k_tokens=config["projector"]["num_tokens"],
        max_seq_len=config["train"]["max_seq_len"],
        input_dim=config["projector"]["input_dim"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"].get("num_workers", 0),
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        imu_llm.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = create_scheduler(
        optimizer=optimizer,
        num_epochs=int(config["train"]["num_epochs"]),
        steps_per_epoch=math.ceil(len(dataset) / config["train"]["batch_size"]),
        warmup_steps=int(config["train"].get("warmup_steps", 0)),
    )
    run_dir = create_run_dir(config)
    save_run_config(run_dir, config)
    writer = create_writer(run_dir)
    imu_llm.to(device)
    projector.to(device)
    grad_accum = max(1, int(config["train"].get("grad_accum", 1)))
    max_grad_norm = float(config["train"].get("max_grad_norm", 1.0))
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(int(config["train"]["num_epochs"])):
        imu_llm.train()
        epoch_loss = 0.0
        step_count = 0
        for batch_idx, batch in enumerate(dataloader, start=1):
            with torch.no_grad():
                imu_embeds = projector(batch["imu_seq"].to(device))
            outputs = imu_llm(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                imu_embeds=imu_embeds,
                imu_range=batch["imu_range"],
            )
            raw_loss = outputs.loss
            loss = raw_loss / grad_accum
            loss.backward()
            should_step = batch_idx % grad_accum == 0 or batch_idx == len(dataloader)
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(imu_llm.parameters(), max_grad_norm).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                grad_norm = 0.0
            loss_value = raw_loss.item()
            epoch_loss += loss_value
            step_count += 1
            ppl = safe_ppl(loss_value)
            writer.add_scalar("train/loss", loss_value, global_step)
            writer.add_scalar("train/ppl", ppl, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)
            if batch_idx % int(config["train"].get("log_every", 10)) == 0:
                print(
                    f"[sft][epoch={epoch}][step={batch_idx}] "
                    f"loss={loss_value:.4f} ppl={ppl:.2f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )
            global_step += 1
        epoch_dir = run_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        imu_llm.llm.save_pretrained(epoch_dir / "llm")
        tokenizer.save_pretrained(epoch_dir / "tokenizer")
        torch.save(projector.state_dict(), epoch_dir / "projector.pt")
        avg_loss = epoch_loss / max(1, step_count)
        save_epoch_metrics(
            run_dir,
            {
                "stage": "sft",
                "epoch": epoch,
                "avg_loss": avg_loss,
                "steps": step_count,
            },
        )
        print(f"[sft][epoch={epoch}] avg_loss={avg_loss:.4f}")
    imu_llm.llm.save_pretrained(run_dir / "llm")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    torch.save(projector.state_dict(), run_dir / "projector.pt")
    writer.close()
    return run_dir
