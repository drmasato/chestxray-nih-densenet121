#!/bin/bash
LOG=/media/morita/ubuntuHDD/chestxray/train_efficientnet.log
TOPIC="chestxray_morita"

echo "EfficientNet-B4 完了を監視中..."
until grep -q "学習完了" "$LOG" 2>/dev/null; do
    sleep 60
done

RESULT=$(grep "Best AUC" "$LOG" | tail -1)

curl -s \
  -H "Title: EfficientNet-B4 学習完了！" \
  -H "Priority: high" \
  -H "Tags: chart_with_upwards_trend" \
  -d "Step 1/5 完了 | $RESULT" \
  ntfy.sh/$TOPIC

echo "通知送信完了"
