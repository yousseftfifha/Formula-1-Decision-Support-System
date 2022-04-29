import React from 'react';

const FormData = () => {
    return (
        <>
            <main className="p-6 sm:p-10 space-y-6">

                <div className="flex flex-col space-y-6 md:space-y-0 md:flex-row justify-between">
                    <div className="mr-6">
                        <h1 className="text-4xl font-semibold mb-2">Data Collection</h1>
                        {/* eslint-disable-next-line react/no-unescaped-entities */}
                        <h2 className="text-gray-600 ml-0.5">Contribute to our cause by sharing your opinion with us </h2>
                    </div>
                </div>

                <section className="grid md:grid-cols-2 xl:grid-cols-4 xl:grid-rows-3 xl:grid-flow-col gap-6">

                    <iframe
                        src="https://docs.google.com/forms/d/e/1FAIpQLSeTYL2u82kfmAv1NSItpgi1tV09mafm47Jb_J19lEWd8SwbAg/viewform?embedded=true"
                        width="1700"
                        height="900"
                        frameBorder="0"
                    />
                </section>

            </main>

        </>
    );
};

export default FormData;
