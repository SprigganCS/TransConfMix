from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Exemplos de uso:
# 1) Modo padrão (ignora JSON e monta tudo em train):
#    python3 montar_set.py
#
# 2) Modo sample (usa JSON para separar train/val):
#    python3 montar_set.py --sample
#
# Observação:
# - Para trocar o conjunto traduzido, altere apenas CAMINHO_TRADUZIDAS.
# - As labels são sempre lidas de DEFAULT_LABELS_SOURCE (caminho estático).

SOURCE_DATASET="sim10k" # "dolphins" or "sim10k"
CAMINHO_TRADUZIDAS = "SET_EXPERIMENTS/official_FULL_sim10k2nuscenes_NOCROP_NOLOAD_NOLAMBDA_PLATEAU"

BASE_CONFMIX = Path("/home/andremedeiros/experiments/ConfMix")

DATASET_NAME = SOURCE_DATASET.strip().lower()
if DATASET_NAME not in {"sim10k", "dolphins"}:
	raise ValueError("SOURCE_DATASET deve ser 'sim10k' ou 'dolphins'")

if DATASET_NAME == "dolphins":
	DEFAULT_SPLIT_JSON = Path("/home/andremedeiros/datasets/samples/dolphins/copied_files.json")
	DEFAULT_IMAGES_SOURCE = BASE_CONFMIX / "SET_EXPERIMENTS" / CAMINHO_TRADUZIDAS
	DEFAULT_LABELS_SOURCE = Path("/home/andremedeiros/datasets/dolphins3_vehicle/labels")
	DEFAULT_DATASET_ROOT = BASE_CONFMIX / "SET_EXPERIMENTS" / CAMINHO_TRADUZIDAS
else:
	DEFAULT_SPLIT_JSON = Path("/home/andremedeiros/datasets/samples/sim10k/copied_files.json")
	DEFAULT_IMAGES_SOURCE = BASE_CONFMIX / CAMINHO_TRADUZIDAS
	DEFAULT_LABELS_SOURCE = Path("/home/andremedeiros/datasets/sim10k/labels")
	DEFAULT_DATASET_ROOT = BASE_CONFMIX / CAMINHO_TRADUZIDAS
DEFAULT_USE_JSON = False


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Monta o dataset no formato YOLO. Por padrão, copia tudo para train. "
			"Com --sample, usa train/val do JSON."
		)
	)
	parser.add_argument(
		"--sample",
		action="store_true",
		default=DEFAULT_USE_JSON,
		help="Modo sample: usa o JSON para dividir em train/val.",
	)
	return parser.parse_args()


def load_split(split_json: Path) -> dict[str, list[str]]:
	if not split_json.exists():
		raise FileNotFoundError(f"JSON não encontrado: {split_json}")
	with split_json.open("r", encoding="utf-8") as file:
		data = json.load(file)

	for key in ("train", "val"):
		if key not in data:
			raise KeyError(f"Chave ausente no JSON: '{key}'")
		if not isinstance(data[key], list):
			raise TypeError(f"A chave '{key}' deve ser uma lista")
	return data


def index_fake_images(images_source: Path) -> dict[str, Path]:
	if not images_source.exists():
		raise FileNotFoundError(f"Pasta de imagens não encontrada: {images_source}")

	fake_map: dict[str, Path] = {}
	for path in images_source.rglob("*_fake.*"):
		stem = path.stem
		if not stem.endswith("_fake"):
			continue
		base_id = stem[: -len("_fake")]
		if base_id not in fake_map:
			fake_map[base_id] = path
	return fake_map


def find_label(labels_source: Path, base_id: str) -> Path | None:
	direct = labels_source / f"{base_id}.txt"
	if direct.exists():
		return direct

	for split in ("train", "val"):
		candidate = labels_source / split / f"{base_id}.txt"
		if candidate.exists():
			return candidate
	return None


def copy_file(src: Path, dst: Path) -> bool:
	if dst.exists():
		return False
	dst.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src, dst)
	return True


def move_image_file(src: Path, dst: Path) -> tuple[bool, bool]:
	dst.parent.mkdir(parents=True, exist_ok=True)

	if dst.exists():
		if src.exists() and src.resolve() != dst.resolve():
			src.unlink()
			return False, True
		return False, False

	shutil.move(str(src), str(dst))
	return True, False


def build_dataset(
	split_data: dict[str, list[str]],
	fake_map: dict[str, Path],
	labels_source: Path,
	dataset_root: Path,
	target_splits: tuple[str, ...],
) -> int:
	missing_images: list[str] = []
	missing_labels: list[str] = []
	moved_images = 0
	cleaned_loose_images = 0
	copied_labels = 0

	for split in target_splits:
		images_dst = dataset_root / "images" / split
		labels_dst = dataset_root / "labels" / split
		images_dst.mkdir(parents=True, exist_ok=True)
		labels_dst.mkdir(parents=True, exist_ok=True)

		for file_name in split_data[split]:
			base_id = Path(file_name).stem

			image_src = fake_map.get(base_id)
			if image_src is None:
				missing_images.append(base_id)
				continue

			label_src = find_label(labels_source, base_id)
			if label_src is None:
				missing_labels.append(base_id)
				continue

			image_dst = images_dst / f"{base_id}{image_src.suffix.lower()}"
			label_dst = labels_dst / f"{base_id}.txt"

			moved, cleaned = move_image_file(image_src, image_dst)
			if moved:
				moved_images += 1
			if cleaned:
				cleaned_loose_images += 1
			if copy_file(label_src, label_dst):
				copied_labels += 1

	print(f"Imagens movidas: {moved_images}")
	print(f"Imagens soltas removidas: {cleaned_loose_images}")
	print(f"Labels copiados: {copied_labels}")

	if missing_images:
		print(f"Imagens _fake não encontradas: {len(missing_images)}")
		print(
			"Exemplos de imagens ausentes: "
			+ ", ".join(missing_images[:10])
			+ (" ..." if len(missing_images) > 10 else "")
		)

	if missing_labels:
		print(f"Labels não encontrados: {len(missing_labels)}")
		print(
			"Exemplos de labels ausentes: "
			+ ", ".join(missing_labels[:10])
			+ (" ..." if len(missing_labels) > 10 else "")
		)

	if missing_images or missing_labels:
		return 1
	return 0


def main() -> int:
	args = parse_args()
	fake_map = index_fake_images(DEFAULT_IMAGES_SOURCE)

	if args.sample:
		split_data = load_split(DEFAULT_SPLIT_JSON)
		required_ids = {
			Path(file_name).stem
			for split in ("train", "val")
			for file_name in split_data.get(split, [])
		}
		available_ids = set(fake_map)
		missing_for_sample = sorted(required_ids - available_ids)

		if not available_ids:
			print("ERRO: nenhuma imagem *_fake foi encontrada na pasta de origem.")
			print(
				"Isso normalmente acontece após rodar o modo padrão, que move as imagens "
				"para images/train."
			)
			print("Restaure as imagens na origem antes de usar --sample.")
			return 2

		if missing_for_sample:
			print(
				"ERRO: o modo --sample exige imagens específicas do JSON, "
				"mas algumas não estão mais na origem."
			)
			print(f"Imagens faltantes para --sample: {len(missing_for_sample)}")
			print(
				"Exemplos ausentes: "
				+ ", ".join(missing_for_sample[:10])
				+ (" ..." if len(missing_for_sample) > 10 else "")
			)
			print("Restaure as imagens na origem antes de usar --sample.")
			return 2

		target_splits = ("train", "val")
	else:
		split_data = {"train": [f"{base_id}.jpg" for base_id in sorted(fake_map)], "val": []}
		target_splits = ("train",)

	return build_dataset(
		split_data=split_data,
		fake_map=fake_map,
		labels_source=DEFAULT_LABELS_SOURCE,
		dataset_root=DEFAULT_DATASET_ROOT,
		target_splits=target_splits,
	)


if __name__ == "__main__":
	raise SystemExit(main())
