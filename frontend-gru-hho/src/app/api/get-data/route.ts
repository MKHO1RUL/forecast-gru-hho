import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const pair = searchParams.get("pair");

  if (!pair) {
    return NextResponse.json(
      { detail: "Parameter 'pair' wajib diisi. Contoh: 'EURUSD=X'" },
      { status: 400 }
    );
  }

  try {
    const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(
      pair
    )}?range=5y&interval=1d&includePrePost=false`;

    const res = await fetch(yahooUrl, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
      next: { revalidate: 3600 },
    });

    if (!res.ok) {
      return NextResponse.json(
        { detail: `Gagal mengambil data dari Yahoo Finance untuk pair: ${pair}. Status: ${res.status}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    const result = data?.chart?.result?.[0];

    if (!result || !result.timestamp || !result.indicators?.quote?.[0]?.close) {
      return NextResponse.json(
        { detail: `Tidak ada data historis ditemukan untuk pair: ${pair}` },
        { status: 404 }
      );
    }

    const timestamps: number[] = result.timestamp;
    const closes: (number | null)[] = result.indicators.quote[0].close;

    const formattedData: { Tanggal: string; Terakhir: number }[] = [];

    for (let i = 0; i < timestamps.length; i++) {
      const closeVal = closes[i];
      if (closeVal !== null && closeVal !== undefined && !isNaN(closeVal)) {
        const dateObj = new Date(timestamps[i] * 1000);
        const yyyy = dateObj.getUTCFullYear();
        const mm = String(dateObj.getUTCMonth() + 1).padStart(2, "0");
        const dd = String(dateObj.getUTCDate()).padStart(2, "0");
        formattedData.push({
          Tanggal: `${yyyy}-${mm}-${dd}`,
          Terakhir: Number(closeVal),
        });
      }
    }

    if (formattedData.length === 0) {
      return NextResponse.json(
        { detail: `Tidak ada data valid untuk pair: ${pair}` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      message: `Data untuk ${pair} berhasil dimuat.`,
      data: formattedData,
      max_batch_size: Math.floor(formattedData.length * 0.7),
    });
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Error fetching data: ${msg}` },
      { status: 500 }
    );
  }
}
