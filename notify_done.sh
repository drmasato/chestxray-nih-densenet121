#!/bin/bash
# 学習完了を監視してiPhoneに通知
LOG=/media/morita/ubuntuHDD/chestxray/train.log
TOPIC="chestxray_morita"

echo "学習完了を監視中..."
until grep -q "学習完了" "$LOG" 2>/dev/null; do
    sleep 60
done

# ログから結果を取得
RESULT=$(grep "Best AUC" "$LOG" | tail -1)

# iPhoneに通知
curl -s \
  -H "Title: 胸部X線学習完了！" \
  -H "Priority: high" \
  -H "Tags: white_check_mark" \
  -d "DenseNet-121 30エポック完了 | $RESULT" \
  ntfy.sh/$TOPIC

echo "通知送信完了"
