import os
import json
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box as shapely_box
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from typing import List, Dict, Any, Tuple, Generator
import logging
from src.utils.logger import setup_logger

class ImageTiler:
    """
    Satellite image tiling pipeline for disaster detection.
    Splits large images into smaller patches while maintaining metadata and annotations.
    """
    def __init__(self, tile_size: int = 512, overlap: int = 0):
        self.tile_size = tile_size
        self.overlap = overlap
        self.logger = setup_logger(log_file='experiments/tiling.log', level='INFO')
        self.metadata_mapping = {}

    def load_images(self, raw_dir: str) -> Generator[Tuple[str, str], None, None]:
        """
        Efficiently yields image paths and their split type from the dataset directory.
        Uses a generator for handling large datasets without loading everything into memory.
        """
        splits = ['train', 'val', 'test']
        for split in splits:
            img_dir = os.path.join(raw_dir, split, 'images')
            if not os.path.exists(img_dir):
                self.logger.warning(f"Split {split} not found in {raw_dir}")
                continue
            
            for img_file in os.listdir(img_dir):
                if img_file.endswith('.png'):
                    yield os.path.join(img_dir, img_file), split

    def tile_image(self, img_path: str) -> List[Dict[str, Any]]:
        """
        Splits a single image into smaller tiles.
        Returns a list of tile data including the cropped Image object and coordinates.
        """
        img = Image.open(img_path)
        w, h = img.size
        img_basename = os.path.basename(img_path).split('.')[0]
        
        tiles_data = []
        stride = self.tile_size - self.overlap
        
        for y in range(0, h - self.overlap, stride):
            for x in range(0, w - self.overlap, stride):
                # Calculate tile boundaries
                right = min(x + self.tile_size, w)
                bottom = min(y + self.tile_size, h)
                left = right - self.tile_size if right == w else x
                top = bottom - self.tile_size if bottom == h else y
                
                tile_img = img.crop((left, top, right, bottom))
                
                tiles_data.append({
                    'image': tile_img,
                    'x': left,
                    'y': top,
                    'w': self.tile_size,
                    'h': self.tile_size,
                    'original_name': img_basename,
                    'original_size': (w, h)
                })
                
        return tiles_data

    def save_tiles(self, tiles_data: List[Dict[str, Any]], output_dir: str):
        """
        Saves tiled images and maintains a metadata mapping file.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for tile in tiles_data:
            tile_name = f"{tile['original_name']}_tile_{tile['x']}_{tile['y']}.png"
            tile_path = os.path.join(output_dir, tile_name)
            
            # Save the tile image
            tile['image'].save(tile_path)
            
            # Update metadata mapping
            if tile['original_name'] not in self.metadata_mapping:
                self.metadata_mapping[tile['original_name']] = {
                    'original_size': tile['original_size'],
                    'tiles': []
                }
            
            self.metadata_mapping[tile['original_name']]['tiles'].append({
                'tile_name': tile_name,
                'x': tile['x'],
                'y': tile['y'],
                'size': (tile['w'], tile['h'])
            })

    def tile_annotations(self, label_path: str, output_dir: str, img_size: Tuple[int, int]):
        """
        Splits xBD JSON annotations into patches corresponding to the tiled images.
        """
        with open(label_path, 'r') as f:
            data = json.load(f)
            
        w, h = img_size
        img_basename = os.path.basename(label_path).split('.')[0]
        stride = self.tile_size - self.overlap
        
        for y in range(0, h - self.overlap, stride):
            for x in range(0, w - self.overlap, stride):
                right = min(x + self.tile_size, w)
                bottom = min(y + self.tile_size, h)
                left = right - self.tile_size if right == w else x
                top = bottom - self.tile_size if bottom == h else y
                
                tile_poly = shapely_box(left, top, right, bottom)
                tile_features = []
                
                for feat in data.get('features', {}).get('xy', []):
                    wkt = feat.get('wkt', '')
                    if not wkt: continue
                    
                    poly = wkt_loads(wkt)
                    if not poly.intersects(tile_poly):
                        continue
                        
                    clipped_poly = poly.intersection(tile_poly)
                    if clipped_poly.is_empty or clipped_poly.area < 1.0:
                        continue
                        
                    def shift_coords(coords):
                        return [(cx - left, cy - top) for cx, cy in coords]
                    
                    if clipped_poly.geom_type == 'Polygon':
                        shifted_poly = Polygon(
                            shift_coords(clipped_poly.exterior.coords),
                            [shift_coords(ring.coords) for ring in clipped_poly.interiors]
                        )
                    elif clipped_poly.geom_type == 'MultiPolygon':
                        shifted_polys = []
                        for p in clipped_poly.geoms:
                            shifted_polys.append(Polygon(
                                shift_coords(p.exterior.coords),
                                [shift_coords(ring.coords) for ring in p.interiors]
                            ))
                        from shapely.geometry import MultiPolygon
                        shifted_poly = MultiPolygon(shifted_polys)
                    else:
                        continue

                    new_feat = feat.copy()
                    new_feat['wkt'] = wkt_dumps(shifted_poly)
                    tile_features.append(new_feat)
                
                if tile_features:
                    tile_data = data.copy()
                    tile_data['features']['xy'] = tile_features
                    tile_label_name = f"{img_basename}_tile_{left}_{top}.json"
                    tile_label_path = os.path.join(output_dir, tile_label_name)
                    with open(tile_label_path, 'w') as f_out:
                        json.dump(tile_data, f_out)

    def process_dataset(self, raw_dir: str, processed_dir: str):
        """
        Executes the full pipeline: loading, tiling, saving, and annotation splitting.
        """
        tiles_base_dir = os.path.join(processed_dir, 'tiles')
        self.metadata_mapping = {} # Reset for new run
        
        for img_path, split in self.load_images(raw_dir):
            split_tiles_dir = os.path.join(tiles_base_dir, split)
            img_dest = os.path.join(split_tiles_dir, 'images')
            lbl_dest = os.path.join(split_tiles_dir, 'labels')
            
            # Ensure directories exist
            os.makedirs(img_dest, exist_ok=True)
            os.makedirs(lbl_dest, exist_ok=True)
            
            # 1. Tile image
            tiles_data = self.tile_image(img_path)
            
            # 2. Save tiles and update mapping
            self.save_tiles(tiles_data, img_dest)
            
            # 3. Handle annotations
            lbl_file = os.path.basename(img_path).replace('.png', '.json')
            lbl_path = os.path.join(os.path.dirname(img_path).replace('images', 'labels'), lbl_file)
            
            if os.path.exists(lbl_path):
                self.tile_annotations(lbl_path, lbl_dest, tiles_data[0]['original_size'])

        # Save the final metadata mapping
        mapping_path = os.path.join(tiles_base_dir, 'tile_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(self.metadata_mapping, f, indent=4)
        
        self.logger.info(f"Tiling complete! Metadata mapping saved at {mapping_path}")

def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Professional Satellite Image Tiling Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--tile_size", type=int, help="Patch size (e.g., 256 or 512)")
    parser.add_argument("--overlap", type=int, help="Pixel overlap between tiles")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    tile_size = args.tile_size or config['preprocessing'].get('tile_size', 512)
    overlap = args.overlap or config['preprocessing'].get('overlap', 0)
    
    tiler = ImageTiler(tile_size=tile_size, overlap=overlap)
    tiler.process_dataset(
        raw_dir=config['data']['xview_dir'],
        processed_dir=config['data']['processed_dir']
    )

if __name__ == "__main__":
    main()
