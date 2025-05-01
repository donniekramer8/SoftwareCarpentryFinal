import xml.etree.ElementTree as ET

shift_x = 100
shift_y = 100

min_x = 0
max_x = 1000
min_y = 0
max_y = 1000


xml_pth = '/home/donald/Desktop/donald/data/GBM_CODA_samples/S22-33022_2A/annotations/S22-33022_2A_0001.xml'
xml_out = '/home/donald/Desktop/donald/data/LC_CODA/Br5417/cropped/tests/S22-33022_2A_0001.xml'

# 1) load
tree = ET.parse(xml_pth)
root = tree.getroot()

# 2) process annotations
for ann in list(root.findall('Annotation')):              # list(...) so we can remove safely
    verts = ann.findall('.//Vertex')
    shifted = []
    drop = False

    # compute shifted coords and check bounds
    for v in verts:
        x_new = float(v.get('X')) + shift_x
        y_new = float(v.get('Y')) + shift_y

        if not (min_x <= x_new <= max_x and min_y <= y_new <= max_y):
            drop = True
            break

        shifted.append((v, x_new, y_new))

    if drop:
        # remove the entire <Annotation> node
        root.remove(ann)
    else:
        # commit the shifted coords back into the XML
        for v, x_new, y_new in shifted:
            v.set('X', str(x_new))
            v.set('Y', str(y_new))

# 3) write out (preserves element/attribute order; whitespace may be normalized)
tree.write(xml_out, encoding='utf-8', xml_declaration=True)