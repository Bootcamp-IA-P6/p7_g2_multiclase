# src/data_loader.py
"""
DataLoader para el dataset de detección de incendios forestales.
Genera train_loader, val_loader y test_loader listos para PyTorch.

Uso:
    from src.data_loader import get_dataloaders, get_class_weights
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import json
import os
from pathlib import Path

import kagglehub
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

load_dotenv()

# ── Configuración global ──────────────────────────────────────
DATASET_HANDLE   = "amerzishminha/forest-fire-smoke-and-non-fire-image-dataset"
DATASET_SUBDIR   = "FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET"
EXCLUDED_PATH    = Path("data/nonfire_excluded.json")
CLASSES          = ["fire", "non fire", "smoke"]
CLASS_TO_IDX     = {cls: idx for idx, cls in enumerate(CLASSES)}
IMG_SIZE         = 224
BATCH_SIZE       = 32
VAL_SPLIT        = 0.15
RANDOM_STATE     = 42

# Valores de normalización ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Dataset personalizado ─────────────────────────────────────
class ForestFireDataset(Dataset):
    """
    Dataset PyTorch para imágenes de incendios forestales.

    Parámetros:
        paths:     lista de rutas de imágenes
        labels:    lista de etiquetas numéricas (0=fire, 1=non fire, 2=smoke)
        transform: transformaciones de torchvision a aplicar
    """

    def __init__(
        self,
        paths:     list,
        labels:    list,
        transform: transforms.Compose = None
    ):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        label    = self.labels[idx]

        # Cargar imagen y convertir siempre a RGB
        # Esto soluciona los 100 casos RGBA, P y L detectados en el EDA
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Si la imagen está corrupta devolver un tensor negro
            print(f"⚠️  Error cargando {img_path}: {e}")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label


# ── Transformaciones ──────────────────────────────────────────
def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Devuelve las transformaciones para train (augment=True)
    o para val/test (augment=False).

    Basado en hallazgos del EDA:
    - Redimensionar a 224x224 (1.826 tamaños únicos detectados)
    - Normalizar con valores ImageNet (transfer learning)
    - Augmentation suave — imágenes médicas/naturales
    """
    base_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    if not augment:
        return transforms.Compose(base_transforms)

    # Augmentation solo para train
    # Suave — no agresivo para no distorsionar patrones de fuego/humo
    augment_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),
        transforms.RandomZoomOut(fill=0, side_range=(1.0, 1.2), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    return transforms.Compose(augment_transforms)


# ── Funciones auxiliares ──────────────────────────────────────
def get_dataset_path() -> Path:
    """
    Descarga el dataset via kagglehub si no está en caché
    o devuelve la ruta local si ya existe.
    Primera vez: ~8GB. Siguientes: instantáneo.
    """
    print("📦 Obteniendo dataset desde kagglehub...")
    path = kagglehub.dataset_download(DATASET_HANDLE)
    path = Path(path) / DATASET_SUBDIR
    print(f"✅ Dataset en: {path}")
    return path


def load_excluded_filenames() -> set:
    """
    Carga la lista de imágenes de non fire excluidas
    detectadas en el EDA por no contener naturaleza.
    """
    if not EXCLUDED_PATH.exists():
        print("⚠️  No se encontró data/nonfire_excluded.json")
        print("   Continuando sin filtrar imágenes de non fire")
        return set()

    with open(EXCLUDED_PATH) as f:
        data = json.load(f)

    excluded = set(data["filenames"])
    print(f"✅ Cargadas {len(excluded)} imágenes excluidas de non fire")
    return excluded


def collect_paths_and_labels(
    split_dir:         Path,
    excluded_filenames: set
) -> tuple:
    """
    Recorre las carpetas de un split y devuelve
    listas paralelas de paths y etiquetas numéricas.
    Filtra las imágenes excluidas del EDA.
    """
    paths  = []
    labels = []

    for cls in CLASSES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            print(f"⚠️  No existe: {cls_dir}")
            continue

        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png", ".bmp"
            ]:
                continue

            # Filtrar imágenes excluidas del EDA
            if img_path.name in excluded_filenames:
                continue

            paths.append(str(img_path))
            labels.append(CLASS_TO_IDX[cls])

    return paths, labels


# ── Función principal ─────────────────────────────────────────
def get_dataloaders(
    batch_size:   int   = BATCH_SIZE,
    val_split:    float = VAL_SPLIT,
    num_workers:  int   = 0,
    seed:         int   = RANDOM_STATE
) -> tuple:
    """
    Devuelve (train_loader, val_loader, test_loader)
    listos para usar en model.fit() de PyTorch.

    Parámetros:
        batch_size:  tamaño del batch (default: 32)
        val_split:   proporción de train para validación (default: 0.15)
        num_workers: workers para carga paralela (0 en Windows)
        seed:        semilla para reproducibilidad

    Retorna:
        train_loader: con augmentation activado
        val_loader:   sin augmentation, shuffle=False
        test_loader:  sin augmentation, shuffle=False
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Obtener rutas del dataset
    dataset_path = get_dataset_path()
    train_dir    = dataset_path / "train"
    test_dir     = dataset_path / "test"

    # Cargar exclusiones del EDA
    excluded = load_excluded_filenames()

    # Recopilar paths y etiquetas
    print("\n⏳ Recopilando paths del dataset...")
    all_train_paths,  all_train_labels  = collect_paths_and_labels(
        train_dir, excluded
    )
    test_paths, test_labels = collect_paths_and_labels(
        test_dir, excluded
    )

    # Split estratificado train/val
    # Estratificado para mantener proporciones de clase (hallazgo EDA)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_paths,
        all_train_labels,
        test_size=val_split,
        stratify=all_train_labels,
        random_state=seed
    )

    print(f"\n✅ División del dataset:")
    print(f"   Train:      {len(train_paths):,} imágenes")
    print(f"   Validación: {len(val_paths):,} imágenes")
    print(f"   Test:       {len(test_paths):,} imágenes")

    # Verificar balance por clase
    print(f"\n   Balance por clase (train):")
    for cls, idx in CLASS_TO_IDX.items():
        n = train_labels.count(idx)
        print(f"   {cls:<12}: {n:,} ({n/len(train_labels)*100:.1f}%)")

    # Crear datasets
    train_dataset = ForestFireDataset(
        train_paths, train_labels,
        transform=get_transforms(augment=True)
    )
    val_dataset = ForestFireDataset(
        val_paths, val_labels,
        transform=get_transforms(augment=False)
    )
    test_dataset = ForestFireDataset(
        test_paths, test_labels,
        transform=get_transforms(augment=False)
    )

    # Crear dataloaders
    # num_workers=0 en Windows para evitar errores de multiprocessing
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle solo en train
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # nunca shuffle en val/test
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # crítico para métricas correctas
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\n✅ DataLoaders creados")
    print(f"   Batch size: {batch_size}")
    print(f"   Batches train: {len(train_loader)}")
    print(f"   Batches val:   {len(val_loader)}")
    print(f"   Batches test:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


def get_class_weights(
    train_loader: DataLoader
) -> torch.Tensor:
    """
    Calcula class weights para compensar el desbalanceo
    detectado en el EDA (smoke tiene menos imágenes).

    Uso en entrenamiento:
        weights  = get_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    """
    labels = train_loader.dataset.labels
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )
    weights_tensor = torch.FloatTensor(weights)

    print("✅ Class weights calculados:")
    for cls, w in zip(CLASSES, weights_tensor):
        print(f"   {cls:<12}: {w:.4f}")

    return weights_tensor


def get_dataset_info() -> dict:
    """
    Devuelve información del dataset para logging
    y para los tests unitarios de Persona D.
    """
    return {
        "classes":      CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "img_size":     IMG_SIZE,
        "batch_size":   BATCH_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std":  IMAGENET_STD,
        "n_classes":    len(CLASSES)
    }


# ── Test rápido ───────────────────────────────────────────────
if __name__ == "__main__":
    print("🔥 Testeando data_loader.py\n")

    train_loader, val_loader, test_loader = get_dataloaders()

    # Verificar shape de un batch
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"\n✅ Shape batch imágenes: {batch_imgs.shape}")
    print(f"✅ Shape batch labels:   {batch_labels.shape}")
    print(f"✅ Tipo tensor:          {batch_imgs.dtype}")
    print(f"✅ Rango valores:        [{batch_imgs.min():.2f}, {batch_imgs.max():.2f}]")
    print(f"✅ Clases en el batch:   {batch_labels.unique()}")

    # Verificar class weights
    weights = get_class_weights(train_loader)
    print(f"\n✅ Class weights: {weights}")

    # Info del dataset
    info = get_dataset_info()
    print(f"\n✅ Info dataset: {info}")

    print("\n✅ data_loader.py funcionando correctamente")