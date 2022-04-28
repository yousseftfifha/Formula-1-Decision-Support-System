import React from 'react';

const Dashboard = () => {
    return (
        <>
        <main className="p-6 sm:p-10 space-y-6">
        
        <div className="flex flex-col space-y-6 md:space-y-0 md:flex-row justify-between">
          <div className="mr-6">
            <h1 className="text-4xl font-semibold mb-2">Dashboard</h1>
              {/* eslint-disable-next-line react/no-unescaped-entities */}
            <h2 className="text-gray-600 ml-0.5">We will make data talk if you're willing to listen</h2>
          </div>
        </div>

        <section className="grid md:grid-cols-2 xl:grid-cols-4 xl:grid-rows-3 xl:grid-flow-col gap-6">

            <iframe
                title="F1"
                width="1700"
                height="900"
                src="https://app.powerbi.com/reportEmbed?reportId=f831feb3-5179-464b-b2f0-7e1e7fcdb8c5&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLW5vcnRoLWV1cm9wZS1pLXByaW1hcnktcmVkaXJlY3QuYW5hbHlzaXMud2luZG93cy5uZXQvIn0%3D"
                frameBorder="0"
                allowFullScreen
            />
        </section>
       
      </main>
      
        </>
    );
};

export default Dashboard;
