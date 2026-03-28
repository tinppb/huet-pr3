import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# 1. Load và tiền xử lý
df_trend = pd.read_csv('nhom3_output/trend_monthly.csv')
df_trend['year_month'] = pd.to_datetime(df_trend['year_month'])

# 2. Lọc dữ liệu từ năm 2015 trở đi
df_filtered = df_trend[df_trend['year_month'] >= '2015-01-01'].copy()

# Thiết lập phong cách
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(16, 7)) # Tăng chiều rộng để thoáng hơn

# 3. Vẽ cột cho Số lượng Review (Trục Y bên trái)
# Chuyển x sang dạng string để seaborn hiểu là phân loại, giúp cột đều nhau
sns.barplot(data=df_filtered, 
            x=df_filtered['year_month'].dt.strftime('%m-%Y'), 
            y='total_reviews', 
            ax=ax1, color='lightblue', alpha=0.7)

ax1.set_ylabel('Số lượng đánh giá', fontsize=12, fontweight='bold', color='darkblue')
ax1.set_xlabel('Tháng - Năm', fontsize=12)

# 4. Tạo trục Y thứ hai cho Điểm Sao (Trục Y bên phải)
ax2 = ax1.twinx()
# Dùng range(len) để khớp vị trí các cột của bar chart
sns.lineplot(x=range(len(df_filtered)), 
             y=df_filtered['avg_star'], 
             ax=ax2, color='red', marker='o', markersize=6, linewidth=2, label='Điểm sao TB')

ax2.set_ylabel('Điểm sao trung bình (1-5)', fontsize=12, fontweight='bold', color='red')
ax2.set_ylim(0, 5.5)

# 5. XỬ LÝ TRỤC X: Chỉ hiện tối đa 15 nhãn để không bị dày đặc
ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))

# Tự động xoay ngày tháng cho đẹp
plt.gcf().autofmt_xdate()

plt.title('XU HƯỚNG CHẤT LƯỢNG DỊCH VỤ KHÁCH SẠN TẠI HUẾ (2015 - 2024)', fontsize=16, fontweight='bold')
plt.tight_layout()

# Lưu ảnh chất lượng cao để dán báo cáo
# plt.savefig('trend_2015_onwards.png', dpi=300)
plt.show()