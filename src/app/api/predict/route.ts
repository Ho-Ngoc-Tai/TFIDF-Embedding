import { NextResponse } from "next/server";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

export async function POST(request: Request) {
  let payload: unknown;

  try {
    payload = await request.json();
  } catch{
    return NextResponse.json(
      { error: "Payload phải là JSON hợp lệ" },
      { status: 400 },
    );
  }

  if (!payload || typeof (payload as { text?: string }).text !== "string") {
    return NextResponse.json(
      { error: "Thiếu trường text" },
      { status: 400 },
    );
  }

  const baseUrl = process.env.ML_SERVICE_URL ?? DEFAULT_BASE_URL;

  try {
    const response = await fetch(`${baseUrl.replace(/\/$/, "")}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { error: data?.detail ?? "ML service trả về lỗi" },
        { status: response.status },
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("[api/predict]", error);
    return NextResponse.json(
      { error: "Không kết nối được tới ML service" },
      { status: 502 },
    );
  }
}
