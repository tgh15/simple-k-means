from flask import Flask, redirect, request, redirect, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

data_awal = ""
data_hasil = ""
dataframe = pd.DataFrame(pd.read_excel("data dosen.xlsx"))

@app.route('/', methods = ['GET', 'POST'])
def index():
    global dataframe
    dataframe = pd.DataFrame(pd.read_excel("data dosen.xlsx"))
    data = dataframe.head().to_json(orient='records')
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)  
        return redirect(request.url)
    return render_template("index.html", data=data)


@app.route('/hasil')
def hasil():
    df = pd.DataFrame(pd.read_excel("data dosen.xlsx"))
    
    df['Usia.1'].replace(['Cukup Diprioritaskan', 'Prioritas', 'Sangat Prioritas'], [1,2,3], inplace=True)
    df['Pengalaman Kerja'].replace(['Cukup Berpengalaman', 'Berpengalaman', 'Sangat Berpengalaman'], [1,2,3], inplace=True)
    df['Nilai BKD 2 tahun terakhir'].replace(['Mencukupi', 'Baik', 'Baik Sekali', 'Sangat Baik'], [1,2,3,4], inplace=True)
    df['Nilai SKP 2 tahun terakhir'].replace(['Mencukupi', 'Baik', 'Baik Sekali', 'Sangat Baik'], [1,2,3,4], inplace=True)
    df['Nilai kehadiran 2 tahun terakhir'].replace(['Mencukupi', 'Baik', 'Baik Sekali', 'Sangat Baik'], [1,2,3,4], inplace=True)
    df['Spesifikasi'].replace(['Kurang Sesuai', 'Sesuai'], [1,2], inplace=True)
    df['Rasio'].replace(['Tidak Mencukupi', 'Mencukupi'], [1,2], inplace=True)
    df['Kesediaan Dosen Pengganti'].replace(['Tidak Tersedia', 'Tersedia'], [1,2], inplace=True)
    df['Reputasi dan Status PT Tujuan'].replace(['Baik', 'Sangat Baik'], [1,2], inplace=True)

    df_mean = df.drop(["no", "Nama", "Usia"], axis=1)
    av_column = df_mean.mean(axis=1)
    df["Index Rata Rata"] = av_column
    global data_awal 
    data_awal = df.to_json(orient='records')
    plt.scatter(df["Masa Kerja"] , df["Index Rata Rata"], s =10, c = "c", marker = "o", alpha = 1)
    plt.savefig('static/images/scatter.png')
    plt.cla()
    plt.clf()
    plt.close()
    plt.switch_backend('agg')

    df_rm = df.drop(["no", "Nama", "Usia", "Usia.1", "Pengalaman Kerja", "Nilai BKD 2 tahun terakhir", "Nilai SKP 2 tahun terakhir", "Nilai kehadiran 2 tahun terakhir", "Spesifikasi", "Rasio", "Kesediaan Dosen Pengganti", "Reputasi dan Status PT Tujuan"], axis=1)
    df_x = df_rm.iloc[:, 0:2]
    x_array = np.array(df_x)
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_array)
    kmeans = KMeans(n_clusters = 3, random_state=123)
    kmeans.fit(x_scaled)
    df["kluster"] = kmeans.labels_
    global data_hasil
    data_hasil = dataframe.copy()
    data_hasil["kluster"] = df["kluster"]
    data_hasil["kluster"].replace([0,1,2], ["Kurang Direkomendasikan", "Direkomendasikan", "Sangat Direkomendasikan"], inplace=True)

    output = plt.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = df.kluster, marker = "o", alpha = 1, )
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=1 , marker="o")
    plt.title("Hasil Klustering K-Means")
    plt.colorbar(output)
    plt.savefig('static/images/hasil-scatter.png')
    plt.cla()
    plt.clf()
    plt.close()
    plt.switch_backend('agg')
    
    return render_template("hasil.html")


@app.route('/data-awal')
def data_awal():
    return dataframe.to_json(orient='records')

@app.route('/data-hasil')
def data_hasil():
    return data_hasil.to_json(orient='records')

if __name__ == "__main__":
    app.run(debug=False)