import streamlit as st
import os
import shutil
import random
import tempfile
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
from ezdxf import readfile
import ezdxf
from shapely.geometry import Polygon, LineString, MultiPolygon, box, Point
from shapely.affinity import translate, rotate
import pickle
from PIL import Image
import math
import plotly.graph_objects as go

st.set_page_config(layout="wide")

def optimize():

    st.title("🏡Автоматическое проектирование застройки земельных участков")

    uploaded_file = st.file_uploader("🖥️Выберите файл в формате .dxf", type=["dxf"])

    def process_dxf_file(uploaded_file, output_folder):
        try:
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            doc = ezdxf.readfile(tmp_path)
            lwpolylines = [e for e in doc.entities if e.dxftype() == 'LWPOLYLINE']

            if not lwpolylines:
                print(f"No LWPOLYLINE entities found in {uploaded_file.name}")  # Use .name attribute
                return None
            
            all_vertices = []
            
            for polyline in lwpolylines:
                points = polyline.get_points()
                vertices = np.array([(x, y) for x, y, *_ in points])
                all_vertices.append(vertices)
            
            if all_vertices:
                combined_vertices = np.vstack(all_vertices)
                
                original_filename = uploaded_file.name
                output_filename = os.path.splitext(original_filename)[0] + '.npy'
                output_path = os.path.join(output_folder, output_filename)
                
                np.save(output_path, combined_vertices)
                print(f"Saved {len(combined_vertices)} vertices to {output_path}")
                return output_path
            
        except Exception as e:
            print(f"Error processing {uploaded_file.name}: {str(e)}")
            return None
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def clear_directory(directory_path):
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    directories = [
        './app_data/numpy',
        './app_data/png'
    ]

    if st.button('Очистить результаты'):
        for directory in directories:
            clear_directory(directory)

    if uploaded_file is not None:
        for directory in directories:
            clear_directory(directory)
        process_dxf_file(uploaded_file, 'app_data/numpy')


    directory = './app_data/numpy'
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    data_numpy = []

    for file in npy_files:
        filepath = os.path.join(directory, file)
        data = np.load(filepath, allow_pickle=True)
        
        if data.shape[-1] >= 3:
            trimmed_data = data[..., :-3]
        else:
            trimmed_data = data

        data_numpy.append(trimmed_data)
        np.save(filepath, trimmed_data)

    directory = './app_data/numpy'
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    df = pd.DataFrame(columns=['filename', 'polygon_index', 'area', 'num_vertices'])
    if npy_files:
        for file in npy_files:
            filepath = os.path.join(directory, file)
            data = np.load(filepath, allow_pickle=True)
            
            if data.ndim == 2:
                polygons = [data]
            elif data.ndim == 3:
                polygons = data
            else:
                continue
            
            for i, polygon in enumerate(polygons):
                try:
                    polygon = np.atleast_2d(polygon)
                    
                    polygon = polygon[np.isfinite(polygon).all(axis=1)]
                    
                    if len(polygon) >= 3:
                        if not np.array_equal(polygon[0], polygon[-1]):
                            polygon = np.vstack([polygon, polygon[0]])
                        
                        poly_obj = Polygon(polygon)
                        
                        if not poly_obj.is_valid:
                            poly_obj = poly_obj.buffer(0)

                        area = poly_obj.area
                        

                        df = pd.concat([
                            df, 
                            pd.DataFrame([{
                                'filename': file,
                                'polygon_index': i,
                                'area': area,
                                'num_vertices': len(polygon)
                            }])
                        ], ignore_index=True)
                    else:
                        pass

                except Exception as e:
                    continue


        index_dxf = len(df) - 1

        polygon_area = df['area'][index_dxf]
        buildings = {
            '(26, 18)': 468,
            '(18, 18)': 324,
            '(26, 16)': 416,
            '(26, 34)': 884,
            '(28, 16)': 448,
            '(44, 18)': 792,
            '(54, 16)': 864,
            'L-26x18+18x18': 792,
            'L-28x16+18x18': 772,
            'L-26x16+18x18': 740,
            'L-26x16+18x26': 884,
            'L-26x18+16x26': 884,
            'L-26x16+16x28': 864,
        }

        max_density_table = {
            3: 10.0,
            4: 11.8,
            5: 13.3,
            6: 14.5,
            7: 15.5,
            8: 16.4,
            9: 17.1,
            10: 17.8,
            11: 18.3,
            12: 18.8,
            13: 19.2,
            14: 19.6,
            15: 20.0,
            16: 20.3,
            17: 20.6,
            18: 20.9,
            19: 21.1,
            20: 21.3,
            21: 21.5,
            22: 21.7,
            23: 21.9,
            24: 22.1,
            25: 22.2
        }

        setback_rules = {
            (2, 4): 8,
            (5, 8): 8,
            (9, 25): 12
        }

        data = []

        for floor in range(3, 26):
            setback = None
            for (min_floor, max_floor), setback_value in setback_rules.items():
                if min_floor <= floor <= max_floor:
                    setback = setback_value
                    break
            max_density = max_density_table[floor]
            max_living_area = polygon_area * (max_density / 10)
            max_footprint_area = max_living_area / (floor * 0.7)
            for building_type, footprint_area in buildings.items():
                data.append({
                    'polygon_area': polygon_area,
                    'floors': floor,
                    'building_type': building_type,
                    'setback': setback,
                    'footprint_area': footprint_area,
                    'max_living_area': max_living_area,
                    'max_footprint_area': max_footprint_area
                })
        df = pd.DataFrame(data)
        print(df.head())

        def load_model_and_encoder(model_path, encoder_path):
            """Load the saved model and one-hot encoder"""
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            
            return model, encoder

        def preprocess_new_data(new_data, encoder, categorical_cols, numerical_cols):
            """Preprocess new data using the saved one-hot encoder"""
            new_encoded = encoder.transform(new_data[categorical_cols])
            encoded_cols = encoder.get_feature_names_out(categorical_cols)
            
            new_encoded_df = pd.DataFrame(new_encoded, columns=encoded_cols, index=new_data.index)
            
            processed_data = pd.concat([new_data[numerical_cols], new_encoded_df], axis=1)
            
            return processed_data

        MODEL_PATH = './model/catboost_model.pkl'
        ENCODER_PATH = './model/onehot_encoder.pkl'
        model, encoder = load_model_and_encoder(MODEL_PATH, ENCODER_PATH)

        categorical_cols = ['building_type']
        numerical_cols = ['polygon_area', 'floors', 'setback', 'footprint_area',
            'max_living_area', 'max_footprint_area']

        processed_data = preprocess_new_data(df, encoder, categorical_cols, numerical_cols)

        predictions = model.predict(processed_data)

        df['predicted_living_area'] = predictions


        final_data = df

        final_data['living_area'] = final_data['footprint_area'] * final_data['floors']

        final_data = final_data[final_data['living_area'] < final_data['max_living_area']]

        final_data = final_data[(final_data['predicted_living_area'] < final_data['max_living_area']) & (final_data['max_living_area'] - final_data['predicted_living_area'] < 500)].sort_values(['predicted_living_area'], ascending=False).head(10)

        # final_data

        building_types = [
            '(26, 18)',
            '(18, 18)',
            '(26, 16)',
            '(26, 34)',
            '(28, 16)',
            '(44, 18)',
            '(54, 16)',
            'L-26x18+18x18',
            'L-28x16+18x18',
            'L-26x16+18x18',
            'L-26x16+18x26',
            'L-26x18+16x26',
            'L-26x16+16x28',
        ]

        def visualization(polygon_coords, setback, building_type, counter, created_plans_ids):
            polygon = Polygon(polygon_coords)
            inner_polygon = polygon.buffer(-setback, join_style=2)

            def create_building(b_type, position=(0, 0), rotation=0):
                if 'L' in b_type:
                    parts = b_type.split('+')
                    part1 = parts[0].split('x')
                    w1 = int(part1[0].split('-')[1])
                    h1 = int(part1[1])
                    part2 = parts[1].split('x')
                    w2 = int(part2[0])
                    h2 = int(part2[1])
                    
                    building = Polygon([
                        (0, 0), (w1, 0), (w1, h1), 
                        (w2, h1), (w2, h1+h2), (0, h1+h2), (0, 0)
                    ])
                else:
                    dims = b_type.strip('()').split(',')
                    w = int(dims[0])
                    h = int(dims[1])
                    building = box(0, 0, w, h)
                
                building = translate(building, position[0], position[1])
                if rotation != 0:
                    building = rotate(building, rotation, origin=(position[0], position[1]))
                
                return building

            def find_valid_position(building_shape, container):
                min_x, min_y, max_x, max_y = container.bounds
                attempts = 300
                
                for _ in range(attempts):
                    x = random.uniform(min_x, max_x - building_shape.bounds[2])
                    y = random.uniform(min_y, max_y - building_shape.bounds[3])
                    rotation = random.choice(list(range(0, 361, 1)))
                    
                    test_building = create_building(building_type, (x, y), rotation)
                    
                    if container.contains(test_building):
                        return test_building, rotation  # Возвращаем и здание, и угол поворота
                
                return None, 0  # Возвращаем None и угол 0, если не удалось разместить

            building = None
            rotation = 0  # Инициализируем переменную rotation
            
            if not inner_polygon.is_empty:
                if isinstance(inner_polygon, MultiPolygon):
                    container = max(inner_polygon.geoms, key=lambda p: p.area)
                else:
                    container = inner_polygon
                
                building, rotation = find_valid_position(create_building(building_type), container)

            # Создаем изображение только если здание было размещено
            if building is not None:
                fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
                ax.set_facecolor('black')
                ax.set_xlim(0, 80)
                ax.set_ylim(0, 80)

                original_poly = plt.Polygon(polygon_coords, closed=True, fill=False, edgecolor='white', linewidth=1)
                ax.add_patch(original_poly)

                if not inner_polygon.is_empty:
                    if inner_polygon.geom_type == 'Polygon':
                        inner_coords = np.array(inner_polygon.exterior.coords)
                        inner_poly = plt.Polygon(inner_coords, closed=True, fill=False, edgecolor='#00BFFF', linewidth=1)
                        ax.add_patch(inner_poly)
                    elif inner_polygon.geom_type == 'MultiPolygon':
                        for poly in inner_polygon.geoms:
                            inner_coords = np.array(poly.exterior.coords)
                            inner_poly = plt.Polygon(inner_coords, closed=True, fill=False, edgecolor='#00BFFF', linewidth=1)
                            ax.add_patch(inner_poly)

                building_coords = np.array(building.exterior.coords)
                building_poly = plt.Polygon(building_coords, closed=True, fill=True, edgecolor='yellow', facecolor='black', linewidth=1, alpha=0.7)
                ax.add_patch(building_poly)
                print(building_type)

                # Добавляем линию разделения для прямоугольных зданий
                if building_type in ['(26, 34)', '(44, 18)', '(54, 16)']:
                    print(building_type, 'INSIDE IF')
                    # Получаем размеры здания
                    dims = building_type.strip('()').split(',')
                    w = int(dims[0])
                    h = int(dims[1])
                    print(w, h)
                    # Определяем параметры линии в зависимости от типа здания
                    if building_type == '(26, 34)':
                        # Линия параллельна малой стороне (26) на расстоянии 18
                        line_coords_1 = [(-26, 18), (0, 18)]
                    elif building_type == '(44, 18)':
                        # Линия параллельна малой стороне (18) на расстоянии 26
                        line_coords_1 = [(-18, 0), (-18, 18)]
                    elif building_type == '(54, 16)':
                        # Линия параллельна малой стороне (16) на расстоянии 28
                        line_coords_1 = [(-26, 0), (-26, 16)]
                    # print(line_coords_1)
                    # Создаем линию и применяем те же преобразования, что и к зданию
                    line_1 = LineString(line_coords_1)
                    line_1 = translate(line_1, building_coords[0][0], building_coords[0][1])
                    if rotation != 0:
                        line_1 = rotate(line_1, rotation, origin=(building_coords[0][0], building_coords[0][1]))

                    # Рисуем линию
                    x, y = line_1.xy
                    ax.plot(x, y, color='yellow', linewidth=1, alpha=0.7)

                
                # Добавляем линию разделения для L-образных зданий
                if 'L' in building_type:
                    parts = building_type.split('+')
                    part1 = parts[0].split('x')
                    part2 = parts[1].split('x')
                    w1 = int(part1[0].split('-')[1])
                    h1 = int(part1[1])
                    w2 = int(part2[0])
                    h2 = int(part2[1])
                    
                    # Определяем базовые координаты линии (до поворота)
                    if w1 > w2:  # Горизонтальное разделение (основная часть сверху)
                        line_coords = [(0, h1), (w2, h1)]
                    
                    # Создаем линию и применяем те же преобразования, что и к зданию
                    line = LineString(line_coords)
                    line = translate(line, building_coords[0][0], building_coords[0][1])
                    if rotation != 0:
                        line = rotate(line, rotation, origin=(building_coords[0][0], building_coords[0][1]))

                    # Рисуем линию
                    x, y = line.xy
                    ax.plot(x, y, color='yellow', linewidth=1, alpha=0.7 )

                ax.axis('off')
                plt.tight_layout()

                os.makedirs('./app_data/png', exist_ok=True)
                plt.savefig(f'./app_data/png/ready_plan_{counter}.png', bbox_inches='tight', pad_inches=0, dpi=100)
                plt.show()
                plt.close()
                
                print(f"Изображение успешно сохранено в ./app_data/png/ready_plan_{counter}.png")
                created_plans_ids.append(counter)
            else:
                print("Не удалось разместить здание заданного размера в зоне отступа - изображение не создано")

        polygon_coords = data_numpy[index_dxf]

        setbacks = list(final_data['setback'])
        building_types = list(final_data['building_type'])

        counter = 1
        created_plans_ids = []
        for building_type, setback in zip(building_types, setbacks):
            visualization(polygon_coords, setback, building_type, counter, created_plans_ids)
            counter += 1

        # print(app_plans_data)
        # WEB ПРИЛОЖЕНИЕ
        created_plans_ids = [x - 1 for x in created_plans_ids]

        app_plans_data = final_data.iloc[created_plans_ids]

        app_plans_data = app_plans_data[['polygon_area', 'setback', 'floors', 'building_type', 'footprint_area', 'max_living_area', 'living_area']]
        app_plans_data['optimization_percent'] = (app_plans_data['living_area'] / app_plans_data['max_living_area']) * 100
        print(app_plans_data)
        app_plans_data['optimization_percent'] = app_plans_data['optimization_percent'].round(2)
        app_plans_data['max_living_area'] = app_plans_data['max_living_area'].round(2)
        # Force the column to be treated as strings
        # app_plans_data['building_type'] = app_plans_data['building_type'].astype(str)

        def generate_table_data(app_plans_data, index_df):
            categories = ["Площадь участка (м2)", "Размер отступа (м)", "Этажность здания", "Размеры здания (м)", "Площадь пятна застройки (м2)", "Максимальная жилая площадь (м2)", "Фактическая жилая площадь (м2)", "Процент оптимизации (%)"]
            values = app_plans_data.iloc[index_df].tolist()
            print('index_df == ', index_df)
            print('values == ', values)
            return pd.DataFrame({
                "Параметр": categories,
                "Значение": values
            })

        def show_result(app_plans_data):
            st.title("📐Возможные варианты застройки:")
            
            png_dir = "./app_data/png"
            
            png_files = [f for f in os.listdir(png_dir) if f.lower().endswith('.png')]
            png_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            print('PNG FILES:', png_files)
            index_df = 0
            cols = st.columns(3)
            for i, png_file in enumerate(png_files):
                print('INDEX:', i)
                print('PNG FILE:', png_file)
                with cols[i % 3]:
                    with st.container(border=True):
                        # title = os.path.splitext(png_file)[0].replace("_", " ").title()
                        title = f'📌План застройки №{i+1}'
                        st.subheader(title)
                        
                        img_path = os.path.join(png_dir, png_file)
                        try:
                            image = Image.open(img_path)
                            st.image(image, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                        
                        st.markdown("**Характеристики:**")
                        table_data = generate_table_data(app_plans_data, index_df)
                        index_df += 1
                        st.dataframe(
                            table_data,
                            column_config={
                                "Attribute": "Attribute",
                                "Value": "Value"
                            },
                            hide_index=True,
                            use_container_width=True
                        )

        show_result(app_plans_data)

def calculator():
    def validate_input(land_area, building_footprint):
        """Проверка корректности введённых данных"""
        errors = []
        
        if building_footprint > land_area:
            errors.append("❌ Площадь пятна застройки не может превышать общую площадь участка")
        
        if land_area <= 0:
            errors.append("❌ Площадь участка должна быть положительным числом")
        
        if building_footprint <= 0:
            errors.append("❌ Площадь пятна застройки должна быть положительным числом")
        
        return errors

    def calculate_kindergarten(residential_area, is_attached):
        """Расчёт параметров детского сада по МНГП"""
        try:
            places_old = math.ceil(max(50, residential_area / 10000 * 36))
            groups_old = max(4, math.ceil(places_old / 20))
            
            places_new = math.ceil(max(50, residential_area / 10000 * 27))
            groups_new = max(4, math.ceil(places_new / 20))
            
            if is_attached:
                buildings_old = math.ceil(places_old / 150)
                buildings_new = math.ceil(places_new / 150)
            else:
                buildings_old = math.ceil(places_old / 350)
                buildings_new = math.ceil(places_new / 350)
            
            return {
                "old": {"places": places_old, "groups": groups_old, "buildings": buildings_old},
                "new": {"places": places_new, "groups": groups_new, "buildings": buildings_new}
            }
        except Exception as e:
            st.error(f"Ошибка расчёта детского сада: {str(e)}")
            return None

    def calculate_school(residential_area):
        """Расчёт параметров школы по МНГП"""
        try:
            places_old = math.ceil(residential_area / 10000 * 76)
            places_new = math.ceil(residential_area / 10000 * 57)
            return {
                "old": {"places": places_old, "building_area": places_old * 20},
                "new": {"places": places_new, "building_area": places_new * 20}
            }
        except Exception as e:
            st.error(f"Ошибка расчёта школы: {str(e)}")
            return None

    def create_pie_chart(labels, values):
        """Создание круговой диаграммы с проверкой данных"""
        try:
            # Фильтрация нулевых значений
            non_zero = [(label, value) for label, value in zip(labels, values) if value > 0]
            if not non_zero:
                return None
            
            filtered_labels, filtered_values = zip(*non_zero)
            
            fig = go.Figure(data=[go.Pie(
                labels=filtered_labels,
                values=filtered_values,
                hole=0.3,
                textinfo='percent+label',
                textposition='inside',
                marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'])
            )])
            
            fig.update_layout(
                title_text="Распределение площади участка",
                showlegend=True,
                height=500,
                uniformtext_minsize=12,
                uniformtext_mode='hide'
            )
            return fig
        except Exception as e:
            st.error(f"Ошибка создания диаграммы: {str(e)}")
            return None

    def calculator_begin():
        # st.set_page_config(page_title="Калькулятор ТЭП", layout="wide")
        st.title("📊 Калькулятор ТЭП для жилого комплекса")
        
        # Ввод параметров
        with st.expander("⚙️ Основные параметры", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                land_area = st.number_input("Площадь участка (кв.м)", min_value=0.0, value=10000.0, step=0.1)
                building_footprint = st.number_input("Площадь пятна застройки (кв.м)", min_value=0.0, value=2000.0, step=0.1)
                floors = st.number_input("Этажность", min_value=1, value=10)
            with col2:
                commercial_ground_floor = st.radio("1-й этаж под коммерцию?", ["Да", "Нет"], index=0)
                is_attached_kindergarten = st.radio("Детский сад встроенно-пристроенный?", ["Да", "Нет"], index=1)

        # Валидация ввода
        errors = validate_input(land_area, building_footprint)
        if errors:
            for error in errors:
                st.error(error)
            return

        try:
            # Расчет площадей
            if commercial_ground_floor == "Да":
                commercial_area = building_footprint * 0.7
                residential_area = building_footprint * (floors - 1) * 0.7
            else:
                commercial_area = 0
                residential_area = building_footprint * floors * 0.7

            total_sellable_area = commercial_area + residential_area
            
            # Расчет социальных объектов
            kindergarten_data = calculate_kindergarten(residential_area, is_attached_kindergarten == "ДА")
            school_data = calculate_school(residential_area)

            # Визуализация
            st.markdown("---")
            st.subheader("📊 Распределение площади участка")
            
            # Расчет компонентов для диаграммы
            building_area = building_footprint
            parking_area = building_footprint * 0.5
            landscaping_area = land_area * 0.2
            sbp_area = land_area * 0.1
            free_area = max(0, land_area - building_area - parking_area - landscaping_area - sbp_area)
            
            # Данные для диаграммы
            labels = ["Здание", "Парковка", "Озеленение", "СБП", "Свободная площадь"]
            values = [building_area, parking_area, landscaping_area, sbp_area, free_area]
            
            # Создание и отображение диаграммы
            fig = create_pie_chart(labels, values)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Нет данных для отображения диаграммы.")

            # Вывод результатов
            st.markdown("---")
            st.subheader("📈 Результаты расчётов")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Жилая площадь (кв.м)", f"{residential_area:,.2f}")
                st.metric("Коммерческая площадь (кв.м)", f"{commercial_area:,.2f}")
            with col2:
                st.metric("Общая площадь участка (кв.м)", f"{land_area:,.2f}")
                st.metric("Свободная площадь участка (кв.м)", f"{free_area:,.2f}")

            # Вывод данных о социальных объектах
            if kindergarten_data and school_data:
                st.markdown("---")
                st.subheader("🏫 Социальные объекты")
                
                st.write("#### Детские сады")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**По старому МНГП (36 мест/10000 кв.м)**")
                    st.metric("Количество мест", kindergarten_data["old"]["places"])
                    st.metric("Количество групп", kindergarten_data["old"]["groups"])
                    st.metric("Количество зданий", kindergarten_data["old"]["buildings"])
                with col2:
                    st.write("**По новому МНГП (27 мест/10000 кв.м)**")
                    st.metric("Количество мест", kindergarten_data["new"]["places"])
                    st.metric("Количество групп", kindergarten_data["new"]["groups"])
                    st.metric("Количество зданий", kindergarten_data["new"]["buildings"])

                st.write("#### Школы")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**По старому МНГП (76 мест/10000 кв.м)**")
                    st.metric("Количество мест", school_data["old"]["places"])
                    st.metric("Площадь здания (кв.м)", f"{school_data['old']['building_area']:,.2f}")
                with col2:
                    st.write("**По новому МНГП (57 мест/10000 кв.м)**")
                    st.metric("Количество мест", school_data["new"]["places"])
                    st.metric("Площадь здания (кв.м)", f"{school_data['new']['building_area']:,.2f}")

        except Exception as e:
            st.error(f"Произошла ошибка при расчётах: {str(e)}")

    calculator_begin()


# Основной код приложения
tab1, tab2 = st.tabs(["Оптимизация застройки", "Калькулятор ТЭП"])

with tab1:
    optimize()

with tab2:
    calculator()